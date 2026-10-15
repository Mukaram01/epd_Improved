"""Pure scoping, capture-contract and dummy supervision regressions."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import diagnostic_shadow_capture as d
import shadow_mask_paste as s


class DiagnosticCaptureTests(unittest.TestCase):
    def proposal(self):
        return {'schema':d.SCHEMA,'purpose':d.PURPOSE,'implementer':'/root',
                'checkpoint_sha256':s.CHECKPOINT_SHA,'dataset_sha256':s.DATASET_SHA,
                'scene_ids':list(s.SCENES),'controls':list(s.CONTROLS),
                'original_training':{'parent_exit':'UNKNOWN','parent_wall':'UNKNOWN','resource_enforcement':'UNKNOWN'},
                'production_qualified':False,'stage_a2':'BLOCKED_0/2',
                'sources_sha256':{},'evidence_sha256':{},'limits':d.LIMITS}

    def test_diagnostic_scope_never_relaxes_original_gate(self):
        p=self.proposal();d.validate_scope(p)
        for key,value in [('purpose','production'),('checkpoint_sha256','bad'),('scene_ids',['scene_024']),
                          ('controls',['erode']),('production_qualified',True),('stage_a2','2/2')]:
            bad=copy.deepcopy(p);bad[key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):d.validate_scope(bad)
        bad=copy.deepcopy(p);bad['original_training']['parent_exit']=0
        with self.assertRaises(ValueError):d.validate_scope(bad)

    def test_review_missing_rejected_or_self_approved_fails(self):
        p=self.proposal()
        good={'decision':'APPROVE_DIAGNOSTIC_ONLY','reviewer':'/root/independent',
              'qualification_sha256':'proposal','sources_sha256':{},'evidence_sha256':{}}
        d.validate_review(p,'proposal',good)
        for changed in [None,dict(good,decision='REJECT'),dict(good,reviewer='/root'),
                        dict(good,qualification_sha256='drift'),dict(good,sources_sha256={'drift':'x'})]:
            with self.assertRaises(ValueError):d.validate_review(p,'proposal',changed)

    def test_capture_zero_detections_and_identity_drift(self):
        b={'logits':np.empty((0,2,56,56),np.float32),'boxes':np.empty((0,4),np.float32),
           'transformed_boxes':np.empty((0,4),np.float32),'labels':np.empty(0,np.int64),
           'scores':np.empty(0,np.float32),'masks':np.empty((0,1,512,512),np.float32)}
        d.validate_capture(b)
        bad=dict(b,labels=np.array([1],np.int64))
        with self.assertRaises(ValueError):d.validate_capture(bad)
        bad=dict(b,logits=np.full((1,2,56,56),np.nan,np.float32))
        with self.assertRaises(ValueError):d.validate_capture(bad)

    def test_semantic_background_is_class_mapping_not_mask_edit(self):
        raw=np.array([[0,9,7],[8,7,9]],np.uint32)
        normalized=d.scoring_ids(raw,{7,8})
        np.testing.assert_array_equal(normalized,[[0,0,7],[8,7,0]])
        np.testing.assert_array_equal(raw,[[0,9,7],[8,7,9]])

    def test_negative_and_unmatched_targets_preserved(self):
        prob=np.zeros((8,8),np.float32);prob[0,0]=.8
        row=d.metrics(prob>.5,np.zeros((8,8),np.uint32),None)
        self.assertEqual(row['background_spill'],1);self.assertIsNone(row['iou'])
        ids=np.zeros((8,8),np.uint32);ids[2:4,2:4]=7
        row=d.metrics(np.zeros_like(ids,dtype=bool),ids,7)
        self.assertEqual(row['missing_foreground'],4);self.assertEqual(row['iou'],0.)

    def test_frozen_baseline_assignment_zero_and_duplicate_predictions(self):
        ids=np.zeros((8,8),np.uint32);ids[2:4,2:4]=7
        masks=np.zeros((2,1,8,8),np.float32);masks[:,0,2:4,2:4]=.9
        refs,assignment=d.fixed_matching(masks,[0,1],ids,{7})
        self.assertEqual(refs,[7,7]);self.assertEqual(len(assignment),1)
        self.assertEqual(d.fixed_matching(masks[:0],[],ids,{7}),([],{}))

    def test_durable_exclusive_output_and_preserved_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'evidence.json';d.durable_json(p,{'kept':True})
            with self.assertRaises(FileExistsError):d.durable_json(p,{'kept':False})
            self.assertEqual(json.loads(p.read_text()),{'kept':True})
            out=Path(tmp)/'out';out.mkdir()
            result=d.supervise([sys.executable,'-c','raise SystemExit(3)'],out,2,1000000)
            self.assertEqual(result['exit_code'],3);self.assertEqual(result['status'],'BLOCKED_WORKER_FAILURE')
            self.assertTrue((out/'supervision.json').exists());self.assertTrue((out/'stderr.log').exists())

    def test_wall_and_memory_termination_preserve_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for label,command,wall,rss in [
                ('wall',[sys.executable,'-c','import time;time.sleep(5)'],.15,1000000),
                ('memory',[sys.executable,'-c','import time;x=bytearray(32*1024*1024);time.sleep(5)'],3,12000)]:
                out=Path(tmp)/label;out.mkdir()
                r=d.supervise(command,out,wall,rss)
                self.assertEqual(r['status'],'BLOCKED_RESOURCE_LIMIT')
                self.assertEqual(r['resource_limit_reason'],label)
                self.assertGreater(r['wall_seconds'],0)


if __name__=='__main__':unittest.main()
