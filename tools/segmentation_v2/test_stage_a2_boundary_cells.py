"""Synthetic regressions for the read-only 2x2 diagnostic."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stage_a2_boundary_audit as a
import stage_a2_boundary_cells as c


class BoundaryCellsTests(unittest.TestCase):
    def setUp(self):
        self.ids256 = np.zeros((256, 256), np.uint32)
        self.ids512 = np.zeros((512, 512), np.uint32)
        self.ids256[100, 100] = 7
        self.ids512[200, 200] = 7
        self.prob = np.zeros((512, 512), np.float32)
        self.box = np.array([0, 0, 512, 512], np.float32)
        # Four cells: target-only, mixed target/background, all background,
        # and other-only support. Values and ordering deliberately asymmetric.
        self.ids512[2:4, 4:6] = [[7, 7], [0, 0]]
        self.ids512[2:4, 6:8] = [[7, 0], [0, 0]]
        self.ids512[2:4, 10:12] = [[9, 0], [0, 0]]
        self.prob[2:4, 4:6] = [[.75, .5], [.1, .2]]
        self.prob[2:4, 6:8] = [[.6, .7], [.5, .1]]
        self.prob[2:4, 8:10] = [[.8, .9], [.99, .51]]
        self.prob[2:4, 10:12] = [[.6, .5], [.2, .3]]

    def measure(self):
        return c.cells(self.prob, self.box, 7, self.ids256, self.ids512)

    def test_partition_conserves_cells_and_samples(self):
        s, v = self.measure()
        self.assertEqual(s['cells'], 4)
        self.assertEqual(sum(s['categories'].values()), 4)
        self.assertEqual(s['categories']['target_only/no_active_background'], 1)
        self.assertEqual(s['categories']['target_only/active_background'], 1)
        self.assertEqual(s['categories']['all_background/active_background'], 1)
        self.assertEqual(s['categories']['other_only/no_active_background'], 1)
        self.assertEqual(s['active_samples_histogram_1_to_4'], [2, 1, 0, 1])
        self.assertEqual(s['union_inactive_slots'], 8)
        self.assertEqual(sum(s['all_probability_margin_histogram']['counts']), 16)
        self.assertEqual(sum(s['active_probability_margin_histogram']['counts']), 8)
        np.testing.assert_array_equal(v['yx'], [[1,2], [1,3], [1,4], [1,5]])
        np.testing.assert_array_equal(v['ids'][0], [7,7,0,0])
        np.testing.assert_array_equal(v['probabilities'][0], [.75,.5,np.float32(.1),np.float32(.2)])

    def test_roi_odd_alignment_preserves_raw_probabilities(self):
        self.box[:] = [5.9, 3.1, 10.9, 4.9]
        self.prob[3,5] = .8
        s, v = self.measure()
        np.testing.assert_array_equal(v['yx'], [[1,2], [1,4]])
        np.testing.assert_array_equal(v['roi'][0], [False,False,False,True])
        self.assertEqual(v['probabilities'][0,0], .75)  # outside ROI, retained
        self.assertEqual(s['active_samples_histogram_1_to_4'], [1,1,0,0])

    def test_strict_threshold_and_empty(self):
        self.prob[:] = .5
        self.assertEqual(self.measure()[0]['cells'], 0)
        self.prob[2,4] = np.nextafter(np.float32(.5), np.float32(1))
        self.assertEqual(self.measure()[0]['cells'], 1)
        self.assertEqual(self.prob[3,4], .5)

    def test_corrupt_arrays_and_absent_identity_fail(self):
        for bad in [np.nan, np.inf, -.01, 1.01]:
            with self.subTest(bad=bad):
                self.prob[0,0] = bad
                with self.assertRaises(ValueError):
                    self.measure()
        self.prob[0,0] = 0
        self.ids256[:] = 0
        with self.assertRaisesRegex(ValueError, 'IDENTITY'):
            self.measure()
        self.ids256[100,100] = 7
        self.ids512 = self.ids512.astype(np.float32)
        with self.assertRaises(ValueError):
            self.measure()

    def test_retained_coordinate_drift_rejected(self):
        _, arrays = a.instance({'masks': self.prob[None,None],
                               'boxes': self.box[None], 'scores': np.array([.9])},
                              0, 7, self.ids256, self.ids512)
        c.check_coordinates(arrays, arrays)
        changed = {k:v.copy() for k,v in arrays.items()}
        changed['d0_mapped256_background_yx'][0,0] += 1
        with self.assertRaisesRegex(ValueError, 'COORDINATES'):
            c.check_coordinates(arrays, changed)

    def test_archive_absence_unpinned_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.npz'
            self.assertIsNone(c.archive(path, 'a'*64))
            path.write_bytes(b'corrupt')
            for pin in [None, 'a'*64]:
                with self.assertRaises(ValueError):
                    c.archive(path, pin)

    def test_manifest_repin_cannot_authorize_changed_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.npz'
            manifest = Path(tmp) / 'artifact_locations.json'
            pred = {'masks':self.prob[None,None], 'boxes':self.box[None],
                    'scores':np.array([.9],np.float32),'labels':np.array([1],np.int64)}
            np.savez_compressed(path,**pred)
            manifest.write_text(json.dumps([{'path':str(path),'sha256':a.digest(path)}]))
            trusted = a.digest(manifest)
            with patch.dict(c.MANIFEST_SHA, {'synthetic':trusted}):
                self.assertEqual(c.manifest_index('synthetic',manifest)[str(path)],a.digest(path))
                # Preserve binary masks but change the diagnostic probabilities.
                pred['masks'] = np.where(pred['masks'] > .5, .95, .25).astype(np.float32)
                np.savez_compressed(path,**pred)
                manifest.write_text(json.dumps([{'path':str(path),'sha256':a.digest(path)}]))
                with self.assertRaisesRegex(ValueError,'ARCHIVE_MANIFEST_SHA'):
                    c.manifest_index('synthetic',manifest)

    def test_margin_histogram_strict_edges(self):
        hist = c.margin_hist(np.array([0., .25, .5, .75, 1.], np.float32))
        self.assertEqual(hist['counts'], [1,1,0,0,1,0,0,1,1])

    def test_complete_synthetic_run_and_hash_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage, retained = root / 'stage', root / 'retained'
            stage.mkdir()
            retained.mkdir()
            (stage / 'capture').mkdir()
            pred = {'masks': np.stack([self.prob,self.prob])[:,None],
                    'boxes': np.stack([self.box,self.box]),
                    'scores': np.array([.9,.9],np.float32),
                    'labels': np.array([1,1],np.int64)}
            np.savez_compressed(stage / 'predictions.npz', **pred)
            for name in ['result.json','process.json','capture/raw_sha256.json']:
                (stage / name).write_text('{}')
            measured, arrays = [], {}
            for i in (0,1):
                row, v = a.instance(pred,i,7,self.ids256,self.ids512)
                measured.append(row)
                arrays.update(v)
            np.savez_compressed(retained / 'boundary_coordinates.npz', **arrays)
            recorded = {'measurements':[{'reference_identity':7}]*2}
            (stage / 'result.json').write_text(json.dumps(recorded))
            prior = {'current':measured, 'diagnostic_coordinates_sha256':a.digest(retained / 'boundary_coordinates.npz'),
                     'inputs_sha256':{key:a.digest(stage / path) for key,path in
                       [('stage_a2_result','result.json'),('stage_a2_process','process.json'),
                        ('stage_a2_predictions','predictions.npz'),('capture_manifest','capture/raw_sha256.json')]},
                     'decision':'DIAGNOSTIC_ONLY_BLOCKED_NO_PROMOTION',
                     'training_supervisor_evidence':'MISSING_UNRESOLVED',
                     'checkpoint_sha256':a.EXPECTED_CANDIDATE_SHA,'limitations':{}}
            report = retained / 'boundary_report.json'
            report.write_text(json.dumps(prior))
            with patch.object(c,'RETAINED_REPORT_SHA',a.digest(report)), \
                 patch.object(a,'BASES',()), \
                 patch.object(a,'validate_recorded'), \
                 patch.object(a,'read_capture',return_value=(self.ids256,self.ids512,{})):
                output = root / 'new'
                result = c.run(retained,stage,output)
                self.assertEqual([r['cells'] for r in result['current']], [4,4])
                self.assertEqual(set(p.name for p in output.iterdir()),
                                 {'boundary_cells.npz','boundary_cells_report.json'})
                self.assertEqual(result['samples_sha256'],a.digest(output / 'boundary_cells.npz'))
                (stage / 'predictions.npz').write_bytes(b'tamper')
                with self.assertRaisesRegex(ValueError,'SOURCE_SHA'):
                    c.run(retained,stage,root / 'blocked')
                self.assertFalse((root / 'blocked').exists())
                report.write_text('{}')
                with self.assertRaisesRegex(ValueError,'RETAINED_REPORT_SHA'):
                    c.run(retained,stage,root / 'blocked')

    def test_existing_output_and_symlink_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / 'out'
            out.mkdir()
            (out / 'keep').write_text('original')
            for path in [out, root / 'link']:
                if path != out:
                    path.symlink_to(root / 'absent')
                with self.assertRaisesRegex(ValueError, 'OUTPUT_NOT_EXCLUSIVE'):
                    c.run(root / 'absent', root / 'absent', path)
            self.assertEqual((out / 'keep').read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
