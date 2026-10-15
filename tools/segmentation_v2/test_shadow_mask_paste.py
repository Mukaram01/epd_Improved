"""Synthetic only: no checkpoint, scene pixels or model inference."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torchvision.models.detection.roi_heads import paste_masks_in_image
import shadow_mask_paste as s


class ShadowPasteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        logits = torch.linspace(-4, 4, 56*56).reshape(1,1,56,56)
        boxes = torch.tensor([[3.25, 2.75, 19.8, 23.1]])
        return logits, boxes

    def test_original_paste_bitwise_parity_subpixel_clipped_and_empty(self):
        logits, _ = self.fixture()
        for box in [[3.25,2.75,19.8,23.1], [0.,0.,17.2,25.1],
                    [0.,0.,31.,31.], [12.99,14.01,13.5,15.1]]:
            boxes = torch.tensor([box], dtype=torch.float32)
            actual = s.reconstruct(logits,boxes,(32,32))
            expected = paste_masks_in_image(logits.sigmoid(),boxes,(32,32))
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(tuple(s.reconstruct(logits[:0], torch.empty(0,4), (32,32)).shape),(0,1,32,32))

    def test_deterministic_immutable_inputs_and_nontrivial_controls(self):
        logits, boxes = self.fixture()
        before_l, before_b = logits.clone(), boxes.clone()
        baseline = s.reconstruct(logits,boxes,(32,32))
        torch.testing.assert_close(baseline,s.reconstruct(logits,boxes,(32,32)),rtol=0,atol=0)
        for control in s.CONTROLS:
            result = s.reconstruct(logits,boxes,(32,32),control)
            self.assertGreater(float((result-baseline).abs().max()),0.)
        torch.testing.assert_close(logits,before_l,rtol=0,atol=0)
        torch.testing.assert_close(boxes,before_b,rtol=0,atol=0)

    def test_inclusive_extent_off_by_one_and_rounding(self):
        logits = torch.full((1,1,56,56), 20.)
        boxes = torch.tensor([[3.25,3.25,18.8,18.8]])
        base = s.reconstruct(logits,boxes,(32,32)) > .5
        exclusive = s.reconstruct(logits,boxes,(32,32),'exclusive_extent') > .5
        rounded = s.reconstruct(logits,boxes,(32,32),'round_expanded_box') > .5
        self.assertEqual(int(base.sum()),324)
        self.assertEqual(int(exclusive.sum()),289)
        self.assertEqual(int(rounded.sum()),289)

    def test_strict_threshold_roi_and_union_alignment(self):
        prob = np.full((8,8),.5,np.float32)
        prob[3,5] = np.nextafter(np.float32(.5),np.float32(1))
        prob[2,4] = .9  # outside original truncated ROI
        native,mapped = s.project(prob,np.array([5.9,3.1,7.9,6.9],np.float32))
        self.assertEqual(int(native.sum()),1)
        self.assertEqual(np.argwhere(mapped).tolist(),[[1,2]])
        self.assertEqual(float(prob[2,4]),float(np.float32(.9)))

    def test_measurements_native_and_mapped_conserve_and_freeze_identity(self):
        prob = np.zeros((8,8),np.float32)
        prob[2,2] = .9  # target
        prob[2,3] = .9  # other
        prob[3,2] = .9  # bg
        ids = np.zeros((8,8),np.uint32); ids[2,2] = 7; ids[2,3] = 8; ids[4,4] = 7
        ids256 = np.zeros((4,4),np.uint32); ids256[2,2] = 7
        row = s.measure(prob,np.array([0,0,8,8],np.float32),ids,ids256,7,prob)
        self.assertEqual(row['native_full']['background_spill'],1)
        self.assertEqual(row['native_full']['cross_object'],1)
        self.assertEqual(row['native_full']['missing_foreground'],1)
        self.assertEqual(row['native_full']['iou'],.25)
        self.assertEqual(row['mapped_union']['background_spill'],1)
        self.assertEqual(row['union_occupancy_1_to_4'],[0,0,1,0])
        self.assertEqual(row['native_changed_pixels'],0)
        self.assertEqual(row['mapped_changed_cells'],0)

    def test_corrupt_logits_boxes_controls_and_shape_fail(self):
        logits,boxes = self.fixture()
        for bad in [float('nan'),float('inf')]:
            changed=logits.clone(); changed[0,0,0,0]=bad
            with self.assertRaises(ValueError): s.reconstruct(changed,boxes,(32,32))
        for box in [[1,2,0,4],[1,2,float('nan'),4]]:
            with self.assertRaises(ValueError): s.reconstruct(logits,torch.tensor([box],dtype=torch.float32),(32,32))
        with self.assertRaises(ValueError): s.reconstruct(logits,boxes,(32,32),'sweep')
        with self.assertRaises(ValueError): s.reconstruct(logits.double(),boxes,(32,32))

    def test_baseline_drift_blocks_before_geometry(self):
        logits,boxes = self.fixture()
        original=s.reconstruct(logits,boxes,(32,32))
        with self.assertRaisesRegex(ValueError,'PARITY'):
            s.parity(original,original+1e-6)
        self.assertEqual(s.parity(original,original)['max_abs'],0.)

    def test_shadow_comparison_gate_and_fixed_reference(self):
        logits,boxes=self.fixture()
        original=paste_masks_in_image(logits.sigmoid(),boxes,(32,32))
        ids=np.zeros((32,32),np.uint32); ids[12:24,3:20]=7
        ids256=np.zeros((16,16),np.uint32); ids256[6:12,2:10]=7
        result=s.compare_shadow(logits,boxes,original,ids,ids256,[7])
        self.assertTrue(result['parity']['bitwise_equal'])
        self.assertEqual(len(result['comparisons']),4)
        self.assertEqual(result['comparisons']['baseline'][0]['native_changed_pixels'],0)
        for control in s.CONTROLS:
            self.assertGreater(result['comparisons'][control][0]['max_abs_probability_change'],0)
        with self.assertRaisesRegex(ValueError,'BASELINE_PARITY'):
            s.compare_shadow(logits,boxes,original+.000001,ids,ids256,[7])

    def test_invalid_fixed_renderer_identity_rejected(self):
        logits,boxes=self.fixture()
        original=paste_masks_in_image(logits.sigmoid(),boxes,(32,32))
        ids=np.zeros((32,32),np.uint32); ids[12:24,3:20]=7
        small=np.zeros((16,16),np.uint32); small[6:12,2:10]=7
        for identity in [0,True,'7',7.9,-1,8]:
            with self.subTest(identity=identity):
                with self.assertRaisesRegex(ValueError,'METRIC_IDENTITY'):
                    s.compare_shadow(logits,boxes,original,ids,small,[identity])

    def test_hash_binding_missing_provenance_and_output_exclusivity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/'epoch.pt'; dataset=root/'dataset.json'
            checkpoint.write_bytes(b'not deserialized')
            dataset.write_text('{}')
            policy={'checkpoint':{'path':str(checkpoint),'sha256':s.digest(checkpoint)},
                    'dataset':{'path':str(dataset),'sha256':s.digest(dataset)},
                    'supervisor':{'path':str(root/'process.json'),'sha256':None}}
            status=s.admission(policy)
            self.assertEqual(status['status'],'BLOCKED_MISSING_TRAINING_PROVENANCE')
            (root/'process.json').write_text('{"exit_code":0}')
            self.assertEqual(s.admission(policy)['status'],'BLOCKED_UNPINNED_TRAINING_PROVENANCE')
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'INPUT_SHA'): s.admission(policy)
            out=root/'out'; out.mkdir(); (out/'keep').write_text('original')
            with self.assertRaisesRegex(ValueError,'OUTPUT_NOT_EXCLUSIVE'):
                s.audit(root/'no-policy',out)
            self.assertEqual((out/'keep').read_text(),'original')

    def test_blocked_audit_publishes_only_admission_and_never_opens_scenes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/'epoch.pt'; dataset=root/'dataset.json'
            checkpoint.write_bytes(b'not a Torch archive')
            rows=[{'scene_id':f'scene_{n:03}','split':'validation','seed':n,
                   'folder':str(root/'DO_NOT_OPEN'),
                   'sha256':{'capture.json.rgb8':str(n)}} for n in range(24,28)]
            dataset.write_text(json.dumps({'records':rows}))
            policy={'checkpoint':{'path':str(checkpoint),'sha256':s.digest(checkpoint)},
                    'dataset':{'path':str(dataset),'sha256':s.digest(dataset)},
                    'supervisor':{'path':str(root/'process.json'),'sha256':None},
                    'scenes':rows,'source_inventory':{},'controls':list(s.CONTROLS),
                    'freeze':{'confidence_threshold':.8,'mask_threshold_strict_gt':.5},
                    'parity_gate':{'probability_bitwise_equal':True,'threshold_differences':0},
                    'limits':{'cpu_threads':2,'max_scenes':4,'wall_seconds':300,'rss_mib':6144}}
            path=root/'policy.json'; path.write_text(json.dumps(policy))
            with patch.object(s,'CHECKPOINT_SHA',s.digest(checkpoint)), \
                 patch.object(s,'DATASET_SHA',s.digest(dataset)), \
                 patch.object(s,'source_inventory',return_value={}):
                out=root/'new'; result=s.audit(path,out)
                self.assertEqual(result['status'],'BLOCKED_MISSING_TRAINING_PROVENANCE')
                self.assertEqual(result['inference_calls'],0)
                self.assertIsNone(result['real_scene_geometry_comparisons'])
                self.assertEqual([p.name for p in out.iterdir()],['admission_report.json'])
                policy['source_inventory']={'drift':True}; path.write_text(json.dumps(policy))
                with self.assertRaisesRegex(ValueError,'SOURCE_VERSION_DRIFT'):
                    s.audit(path,root/'drift')
                self.assertFalse((root/'drift').exists())

    def test_full512_batch_parity_and_out_of_frame_rejection(self):
        logits=torch.linspace(-8,8,2*56*56).reshape(2,1,56,56)
        boxes=torch.tensor([[16.25,78.75,118.8,229.1],[0.,0.,512.,512.]])
        original=paste_masks_in_image(logits.sigmoid(),boxes,(512,512))
        self.assertTrue(s.parity(original,s.reconstruct(logits,boxes,(512,512)))['bitwise_equal'])
        with self.assertRaisesRegex(ValueError,'PASTE_CONTRACT'):
            s.reconstruct(logits[:1],torch.tensor([[33.,34.,36.,38.]]),(32,32))

    def test_preregistration_deny_heldout_acceptance_and_duplicate_scene(self):
        rows=[{'scene_id':f'scene_{n:03}','split':'validation','seed':n,
               'sha256':{'capture.json.rgb8':str(n)}} for n in range(24,28)]
        self.assertEqual(s.select_scenes(rows),rows)
        for bad in [dict(rows[0],split='test'),dict(rows[0],scene_id='stage_a2'),
                    dict(rows[0],sha256={'capture.json.rgb8':s.ACCEPTANCE_RGB_SHA})]:
            with self.assertRaisesRegex(ValueError,'SCENE_POLICY'):
                s.select_scenes([bad]+rows[1:])


if __name__=='__main__': unittest.main()
