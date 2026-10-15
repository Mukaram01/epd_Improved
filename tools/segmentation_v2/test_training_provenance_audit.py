"""Synthetic forensic regressions. No model construction, inference or training."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import numpy as np
import training_provenance_audit as a


class ProvenanceAuditTests(unittest.TestCase):
    def test_history_exact_order_losses_and_sequence(self):
        rows=[{'step':1,'epoch':1,'scene_id':'a','seconds':2.,'loss':5.,'loss_mask':2.,'loss_classifier':1.},
              {'step':2,'epoch':1,'scene_id':'b','seconds':3.,'loss':5.,'loss_mask':2.,'loss_classifier':1.}]
        self.assertEqual(a.check_history(rows,['a','b'])['summed_update_seconds'],5.)
        for changes in [{'step':2},{'scene_id':'b'},{'loss':float('nan')},{'seconds':-1.},{'loss':4.}]:
            bad=[dict(rows[0],**changes),rows[1]]
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError): a.check_history(bad,['a','b'])

    def test_non_mask_state_comparison_includes_buffers(self):
        import torch
        initial={'backbone.weight':torch.tensor([1.]),'backbone.running_var':torch.tensor([2.]),
                 'roi_heads.mask_head.weight':torch.tensor([3.])}
        final={k:v.clone() for k,v in initial.items()}; final['roi_heads.mask_head.weight']+=1
        result=a.compare_states(initial,final)
        self.assertEqual(result['frozen_tensors_equal'],2)
        self.assertEqual(result['mask_tensors_changed'],1)
        final['backbone.running_var']+=1
        with self.assertRaisesRegex(ValueError,'FROZEN_STATE'): a.compare_states(initial,final)
        final.pop('backbone.running_var')
        with self.assertRaises(ValueError): a.compare_states(initial,final)

    def test_checkpoint_embedded_bindings_not_worker_success(self):
        state={'model':{},'binding':{'sources':{}},'policy':{},'epoch':1,'step':300,
               'seed':260110,'architecture':'mask_roi28_logits56',
               'mask_roi_output_size':28,'mask_logits_size':56,
               'history_sha256':'h','validation_sha256':'v'}
        a.check_checkpoint_metadata(state,state['binding'],{},'h','v')
        for key in ['step','history_sha256','validation_sha256','architecture']:
            changed=dict(state);changed[key]=None
            with self.assertRaisesRegex(ValueError,'CHECKPOINT_BINDING'):
                a.check_checkpoint_metadata(changed,state['binding'],{},'h','v')

    def test_worker_evidence_never_synthesizes_supervision(self):
        matrix=a.provenance_matrix(parent_present=False)
        self.assertEqual(matrix['parent_completion']['status'],'MISSING')
        self.assertEqual(matrix['parent_observed_exit_code']['value'],None)
        self.assertEqual(matrix['parent_resource_limit_reason']['value'],None)
        self.assertEqual(matrix['parent_observed_peak_rss_kib']['value'],None)
        self.assertEqual(matrix['worker_completion']['status'],'VERIFIED')
        # Even finding an unreviewed file is not an admission decision.
        found=a.provenance_matrix(parent_present=True)
        self.assertEqual(found['parent_completion']['status'],'PARTIALLY VERIFIED')

    def test_present_wrong_process_is_not_parent_proof(self):
        self.assertFalse(a.matches_parent({'command':['python','stage_a2_mask_check.py','worker','--campaign','/run']},Path('/run')))
        self.assertFalse(a.matches_parent({'command':['python','mask_resolution_campaign.py','worker','--output','/other']},Path('/run')))
        self.assertTrue(a.matches_parent({'command':['python','mask_resolution_campaign.py','worker','--output','/run']},Path('/run')))

    def test_hash_and_existing_output_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'input'; p.write_text('original')
            with self.assertRaisesRegex(ValueError,'INPUT_SHA'): a.require_hash(p,'0'*64)
            out=root/'out';out.mkdir();(out/'keep').write_text('original')
            with self.assertRaisesRegex(ValueError,'OUTPUT_NOT_EXCLUSIVE'):
                a.audit(root/'absent',out)
            self.assertEqual((out/'keep').read_text(),'original')

    def test_real_bounded_parent_can_die_while_detached_worker_publishes(self):
        # Controlled reproduction of lifecycle only. Real unchanged bounded()
        # supervises a dummy writer, never a training/model worker.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            worker=root/'dummy.py'
            worker.write_text("import os,time,json\nfrom pathlib import Path\np=Path(__file__).parent\n(p/'worker_pid').write_text(str(os.getpid()))\ntime.sleep(.4)\n(p/'result.json').write_text(json.dumps({'dummy_worker_completed':True}))\n")
            parent=root/'parent.py'
            parent.write_text("import sys\nfrom pathlib import Path\nfrom pilot import bounded,write_json\np=Path(__file__).parent\nr=bounded([sys.executable,str(p/'dummy.py')],p,10,memory_mib=512)\nwrite_json(p/'process.json',r)\n")
            env=dict(os.environ,PYTHONPATH=str(Path(a.__file__).parent),OPENBLAS_NUM_THREADS='1')
            p=subprocess.Popen([sys.executable,str(parent)],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            worker_pid=None
            try:
                deadline=time.monotonic()+8
                while not (root/'worker_pid').exists() and time.monotonic()<deadline: time.sleep(.01)
                self.assertTrue((root/'worker_pid').exists())
                worker_pid=int((root/'worker_pid').read_text())
                p.terminate();p.wait(timeout=3)
                deadline=time.monotonic()+3
                while not (root/'result.json').exists() and time.monotonic()<deadline: time.sleep(.01)
                self.assertEqual(json.loads((root/'result.json').read_text()),{'dummy_worker_completed':True})
                self.assertFalse((root/'process.json').exists())
            finally:
                if p.poll() is None:p.kill();p.wait()
                if worker_pid:
                    try:os.killpg(worker_pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                p.stderr.close()


if __name__=='__main__':unittest.main()
