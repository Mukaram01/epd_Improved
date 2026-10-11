"""Unit fixtures below are arithmetic-only; never pilot execution evidence."""
import copy
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pilot


class Contracts(unittest.TestCase):
    def test_frozen_splits_and_negatives(self):
        a = pilot.scene_plan()
        self.assertEqual(a, pilot.scene_plan())
        self.assertEqual([sum(s['split'] == k for s in a['scenes'])
                          for k in ('train', 'validation', 'test')], [24, 12, 12])
        self.assertEqual(len({s['seed'] for s in a['scenes']}), 48)
        for split in ('train', 'validation', 'test'):
            self.assertTrue(any(s['negative'] for s in a['scenes'] if s['split'] == split))
        self.assertNotIn('historical', str(a['scenes']))

    def test_strict_metrics_and_threshold(self):
        ids = np.array([[0, 11, 11], [0, 11, 22]], dtype=np.uint32)
        prob = np.array([[.5, 1, 1], [.8, 0, .9]], dtype=np.float32)
        m = pilot.mask_metrics(prob, ids, 11, {11, 22})
        self.assertEqual(m['background_spill'], 1)
        self.assertEqual(m['cross_object'], 1)
        self.assertEqual(m['missing_visible'], 1)
        self.assertFalse(m['strict_accept'])
        self.assertEqual(m['iou'], 2 / 5)
        exact = pilot.mask_metrics((ids == 11).astype('f4'), ids, 11, {11, 22})
        self.assertTrue(exact['strict_accept'])
        self.assertEqual(exact['boundary_mean_px'], 0)

    def test_native_contract_rejects_unknown_ids_and_stale_capture(self):
        r = dict(success=True, width=256, height=256, shared_scene_pass=True,
                 id_format='R32_UINT', depth_test=True, depth_write=True,
                 render_pass_count=1, acquisition=dict(step=2, stamp_ns=2000000),
                 mapping=[dict(renderer_id=7, class_id=1, model='cube_0')],
                 frozen_equal=True)
        ids = np.zeros((256, 256), dtype='<u4'); ids[20:40, 20:40] = 7
        pilot.validate_native(r, ids)
        bad = copy.deepcopy(r); bad['acquisition']['step'] = 3
        with self.assertRaises(ValueError): pilot.validate_native(bad, ids)
        ids[0, 0] = 99
        with self.assertRaises(ValueError): pilot.validate_native(r, ids)

    def test_timeout_preserves_failure_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            result=pilot.bounded(['/bin/sh','-c','echo "failure evidence"; exec sleep 10'],folder,1)
            self.assertEqual(result['exit_code'],124)
            self.assertEqual(result['resource_limit_reason'],'wall_timeout')
            self.assertIn('failure evidence',(folder/'stdout.log').read_text())

    def test_training_refuses_incomplete_acquisition(self):
        import train
        import json
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'acquisition.json').write_text(json.dumps(dict(status='BLOCKED',scenes=[])))
            with self.assertRaisesRegex(ValueError,'48 genuinely'):
                list(train.verified_rows(folder,'train'))

    def test_split_membership_rejects_duplicate_and_reassigned_scenes(self):
        plan=pilot.scene_plan(); acquisition=dict(scenes=copy.deepcopy(plan['scenes']))
        pilot.validate_split_membership(plan,acquisition)
        acquisition['scenes'][24]=copy.deepcopy(acquisition['scenes'][0])
        with self.assertRaisesRegex(ValueError,'split membership'):
            pilot.validate_split_membership(plan,acquisition)
        acquisition=dict(scenes=copy.deepcopy(plan['scenes']))
        acquisition['scenes'][0]['split']='test'
        with self.assertRaisesRegex(ValueError,'split membership'):
            pilot.validate_split_membership(plan,acquisition)

    def test_negative_false_positives_penalize_validation(self):
        import train
        clean=train.selection_score(.8,1,0,12,12)
        extra=train.selection_score(.8,1,4,12,12)
        self.assertLess(extra,clean)
        self.assertEqual(train.selection_score(0,512,3,0,12),-1.25)

    def test_all_negative_predictions_have_background_spill(self):
        import train
        rows=train.prediction_metrics(np.ones((2,3,3),dtype='f4'),np.full((3,3),7,dtype='u4'),set())
        self.assertEqual(len(rows),2)
        self.assertEqual(sum(r['background_spill'] for r in rows),18)
        self.assertFalse(any(r['strict_accept'] for r in rows))


if __name__ == '__main__': unittest.main()
