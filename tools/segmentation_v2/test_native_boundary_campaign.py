"""No campaign/model training: disposable subprocesses and artificial metrics."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import native_boundary_candidate as c
import native_boundary_validation as v

class CandidateTests(unittest.TestCase):
    def test_durable_supervision_completion_and_wall_failure(self):
        for timeout,code in [(2.,'print("dummy")'),(.03,'import time;time.sleep(5)')]:
            with tempfile.TemporaryDirectory() as d:
                out=Path(d)
                p=c.supervise_training([sys.executable,'-c',code],out,timeout,2097152)
                self.assertTrue((out/'process_started.json').exists())
                self.assertEqual(json.loads((out/'process.json').read_text())['status'],p['status'])
                self.assertEqual(p['resource_limit_reason'],None if timeout==2 else 'wall')
                self.assertTrue((out/'stdout.log').exists());self.assertTrue(p['process_group_cleanup'])
    def test_direct_hidden_worker_cannot_bypass_supervisor(self):
        from unittest.mock import patch
        with patch('sys.argv',['candidate','--worker','--authorise-one-campaign','--config','/absent','--review','/absent','--review-sha256','0'*64]):
            with self.assertRaisesRegex(ValueError,'LIVE_SUPERVISOR_REQUIRED'):c.main()

    def test_no_training_without_explicit_authorisation(self):
        from unittest.mock import patch
        with patch('sys.argv',['candidate','--config','/absent','--review','/absent','--review-sha256','0'*64]):
            with self.assertRaisesRegex(ValueError,'AUTHORISATION'):c.main()
    def test_synthetic_validation_materiality_and_no_harm(self):
        metric={'iou':.8,'missing_foreground':0,'cross_object':0}
        baseline={'spill':10,'cross_object':0,'recovered':1,'instances':{'1':metric},'detection_metrics':[metric]}
        improved=copy.deepcopy(baseline);improved['spill']=8
        scene={'negative':False,'confident_detections':1,'baseline':{'native':baseline,'mapped':baseline},'candidate':{'native':improved,'mapped':improved}}
        self.assertEqual(v.accept_scenes([scene])['decision'],'DEVELOPMENT_PASS_NOT_PRODUCTION')
        scene=copy.deepcopy(scene);scene['candidate']['native']['instances']['1']['missing_foreground']=1
        self.assertEqual(v.accept_scenes([scene])['decision'],'REJECTED')
    def test_full_set_cannot_hide_sentinel_materiality_failure(self):
        metric={'iou':.8,'missing_foreground':0,'cross_object':0}
        base={'spill':10,'cross_object':0,'recovered':1,'instances':{'1':metric},'detection_metrics':[metric]}
        new=copy.deepcopy(base);new['spill']=9
        scene={'scene_id':'s','negative':False,'confident_detections':1,'baseline':{'native':base,'mapped':base},'candidate':{'native':new,'mapped':new}}
        other=copy.deepcopy(scene);other['scene_id']='other';other['baseline']['native']['spill']=100;other['baseline']['mapped']['spill']=100
        other['candidate']['native']['spill']=0;other['candidate']['mapped']['spill']=0
        gate={'scene_ids':['s'],'baseline_spill':{'native':10,'mapped':10},'maximum_candidate_spill':{'native':8,'mapped':8}}
        self.assertEqual(v.accept_scenes([scene,other])['decision'],'DEVELOPMENT_PASS_NOT_PRODUCTION')
        self.assertEqual(v.accept_scenes([scene,other],gate)['decision'],'REJECTED')

    def test_unassigned_detection_cannot_collapse(self):
        good={'iou':.8,'missing_foreground':0,'cross_object':0}
        base={'spill':10,'cross_object':0,'recovered':1,'instances':{'1':copy.deepcopy(good)},'detection_metrics':[copy.deepcopy(good),copy.deepcopy(good)]}
        new=copy.deepcopy(base);new['spill']=8;new['detection_metrics'][1].update(iou=0.,missing_foreground=100)
        scene={'negative':False,'confident_detections':2,'baseline':{'native':base,'mapped':base},'candidate':{'native':new,'mapped':new}}
        self.assertEqual(v.accept_scenes([scene])['decision'],'REJECTED')

    def test_prediction_identity_and_negative_false_positive(self):
        b={'boxes':np.empty((0,4),np.float32),'labels':np.empty(0,np.int64),'scores':np.empty(0,np.float32),'masks':np.empty((0,1,512,512),np.float32)}
        scene=v.compare_scene(b,b,np.zeros((512,512),np.uint32),np.zeros((256,256),np.uint32),set())
        self.assertTrue(scene['negative']);self.assertEqual(scene['confident_detections'],0)
        # With no baseline spill, do not invent a 20% material improvement.
        self.assertEqual(v.accept_scenes([scene])['decision'],'REJECTED')

if __name__=='__main__':unittest.main()
