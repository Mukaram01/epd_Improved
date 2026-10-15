"""Prospective admission tests and disposable supervision only; no campaign."""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import foreground_preserving_candidate as c
import test_native_boundary_campaign as historical

class RevisedSupervisorTests(historical.CandidateTests):
 def run(self,result=None):
  with patch.object(historical,'c',c):return super().run(result)

class ScopeTests(unittest.TestCase):
 def test_exact_configuration_and_no_weakened_gates(self):
  old=json.loads((Path(__file__).parent/'native_boundary_candidate.json').read_text())
  new=json.loads((Path(__file__).parent/'foreground_preserving_candidate.json').read_text())
  for k in ['recipe','gate','dataset','baseline','initialization']:self.assertEqual(old[k],new[k])
  self.assertNotEqual(old['output_path'],new['output_path']);self.assertFalse(Path(new['output_path']).exists())
  self.assertEqual(new['loss_revision'],{'name':'foreground_anchor_v1','coefficient':.5})
  self.assertFalse(new['training_authorised']);self.assertEqual(new['base_pr'],229)
 def test_binding_and_loss_identity_drift(self):
  p=json.loads((Path(__file__).parent/'foreground_preserving_candidate.json').read_text())
  c.verify_inputs(p)
  for key,value in [('identity','bad'),('loss_revision',{'name':'foreground_anchor_v1','coefficient':1.})]:
   bad=copy.deepcopy(p);bad[key]=value
   with self.assertRaisesRegex(ValueError,'DRIFT'):c.verify_inputs(bad)
  bad=copy.deepcopy(p);bad['sources_sha256'][str((Path(__file__).parent/'foreground_preserving_aux.py').resolve())]='0'*64
  with self.assertRaisesRegex(ValueError,'SOURCE_DRIFT'):c.verify_inputs(bad)

if __name__=='__main__':unittest.main()
