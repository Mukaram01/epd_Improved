"""Analytic tests; no checkpoints, scene data, trained weights or motion."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torchvision.ops import roi_align
from torchvision.models.detection.roi_heads import paste_masks_in_image
import analytic_coordinate_contract as a
import shadow_mask_paste as s


class AnalyticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_numpy_roi_exact_cpp_parity_fractional_border_and_tiny(self):
        y,x=np.mgrid[:32,:32];field=np.stack([x+y,x*.2-y*.7,np.sin(x/8)+np.cos(y/9)]).astype(np.float32)
        for box in [[2.25,3.75,21.9,25.1],[0,0,31.9,31.9],[.1,.2,.3,.4],[110.25,112.75,128.,128.]]:
            for scale in [1.,.5,.25]:
                for aligned in [False,True]:
                    b=np.array(box,np.float32)
                    ref,grid=a.numpy_roi(field,b,scale,aligned)
                    actual=roi_align(torch.from_numpy(field)[None],torch.tensor([[0,*box]],dtype=torch.float32),28,scale,2,aligned)[0].numpy()
                    np.testing.assert_array_equal(ref,actual)
                    self.assertEqual(grid.shape,(28,28,2,2,2))

    def test_invalid_degenerate_boxes_and_numeric_drift_fail(self):
        f=np.zeros((1,32,32),np.float32)
        for box in [[2,2,2,4],[3,2,1,4],[0,0,np.nan,4]]:
            with self.assertRaises(ValueError):a.numpy_roi(f,np.array(box,np.float32),1.,False)
        with self.assertRaises(ValueError):a.require_exact(np.zeros(1,np.float32),np.array([1e-7],np.float32),'drift')
        with self.assertRaises(ValueError):a.numpy_roi(f.astype(np.float64),np.array([0,0,4,4],np.float32),1.,False)

    def test_analytic_shapes_have_independent_reference_membership(self):
        box=np.array([10.,10.,30.,30.])
        for shape in a.SHAPES:
            self.assertEqual(bool(a.contains(shape,np.array([20.]),np.array([20.]),box)[0]),shape!='empty' and shape!='sharp_edge')
        np.testing.assert_array_equal(a.contains('circle',np.array([20.,26.,27.]),np.array([20.,20.,20.]),box),[True,False,False])
        np.testing.assert_array_equal(a.contains('thin_feature',np.array([20.,20.8]),np.array([20.,20.]),box),[True,False])

    def test_signed_distance_and_conservative_uncertainty(self):
        box=np.array([10.,10.,30.,30.]);x=np.array([20.,26.,28.]);y=np.full(3,20.)
        np.testing.assert_allclose(a.sdf('circle',x,y,box),[6.,0.,-2.],rtol=0,atol=1e-12)
        ref=a.reference('circle',box)
        self.assertGreater(ref['uncertain_pixels'],0)
        self.assertGreater(ref['coverage_area8'],ref['centre_area']-ref['uncertain_pixels'])
        self.assertLessEqual(abs(ref['coverage_area8']-ref['coverage_area4']),ref['uncertain_pixels'])

    def test_continuous_paste_matches_independent_grid_reference(self):
        logits=torch.linspace(-6,6,56*56).reshape(1,1,56,56)
        box=torch.tensor([[10.25,12.75,42.25,40.75]])
        actual=a.continuous_paste(logits,box[0])
        ref=a.numpy_continuous_paste(logits.sigmoid()[0,0].numpy(),box[0].numpy())
        np.testing.assert_allclose(actual,ref,rtol=0,atol=3e-6)
        original=s.reconstruct(logits,box,(128,128))
        torch.testing.assert_close(original,paste_masks_in_image(logits.sigmoid(),box,(128,128)),rtol=0,atol=0)

    def test_empty_strict_threshold_and_conservation(self):
        box=np.array([10.,10.,30.,30.]);ref=a.reference('empty',box)
        p=np.full((128,128),.5,np.float32)
        m=a.measure(p,'empty',box,ref)
        self.assertEqual(m['predicted_foreground'],0);self.assertEqual(m['iou'],1.)
        p[0,0]=np.nextafter(np.float32(.5),np.float32(1))
        m=a.measure(p,'empty',box,ref)
        self.assertEqual(m['spill'],1);self.assertEqual(m['foreground_delta'],m['spill']-m['missing'])

    def test_pixel_centres_half_pixel_offset_and_bridge(self):
        field=np.broadcast_to((np.arange(128)+.5)[None,None,:],(1,128,128)).copy().astype(np.float32)
        b=np.array([16.,16.,44.,44.],np.float32)
        legacy,_=a.numpy_roi(field,b,1.,False);aligned,_=a.numpy_roi(field,b,1.,True)
        np.testing.assert_array_equal(legacy-aligned,np.full_like(legacy,.5))
        pooled=torch.arange(28*28,dtype=torch.float32).reshape(1,1,28,28)
        expected=torch.nn.functional.conv_transpose2d(pooled,torch.ones(1,1,2,2),stride=2)
        torch.testing.assert_close(a.bridge(pooled,'replicate'),expected,rtol=0,atol=0)

    def test_deterministic_clipped_reference_and_snapshot_immutability(self):
        box=np.array([-1.25,-1.5,30.75,26.5]);r=a.reference('rotated_rectangle',box)
        np.testing.assert_array_equal(r['mask'],a.reference('rotated_rectangle',box)['mask'])
        before=box.copy();a.reference('sharp_edge',box);np.testing.assert_array_equal(box,before)

    def test_missing_contour_reports_extinguished_thin_feature(self):
        box=np.array([10.,10.,30.,30.]);ref=a.reference('thin_feature',box)
        result=a.measure(np.zeros((128,128),np.float32),'thin_feature',box,ref)
        self.assertTrue(result['shape_extinguished'])
        self.assertIsNone(result['boundary_abs_mean'])
        self.assertGreater(result['missing'],0)
        self.assertEqual(result['iou'],0.)

    def test_trace_coordinates_tensor_boundary_serializes_without_mutation(self):
        box=torch.tensor([32.25,36.75,64.25,64.75],dtype=torch.float32)
        before=box.clone()
        x,y=a.canonical56_coordinates(box)
        np.testing.assert_array_equal(x,32.25+(np.arange(56)+.5)*32/56)
        np.testing.assert_array_equal(y,36.75+(np.arange(56)+.5)*28/56)
        torch.testing.assert_close(box,before,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.npz'
            a.durable_npz(path,{'canonical56_world_x':x,'canonical56_world_y':y})
            with np.load(path,allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['canonical56_world_x'],x)
                np.testing.assert_array_equal(saved['canonical56_world_y'],y)

    def test_exclusive_output_and_resource_failure_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'keep').write_text('original')
            with self.assertRaisesRegex(ValueError,'OUTPUT_NOT_EXCLUSIVE'):
                a.launch(Path('/absent'),out)
            self.assertEqual((out/'keep').read_text(),'original')
        # Actual time/RSS termination is covered by unchanged shared supervisor tests.


if __name__=='__main__':unittest.main()
