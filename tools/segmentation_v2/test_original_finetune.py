import importlib.util
import json
from pathlib import Path
import unittest
import tempfile


class OriginalFinetuneTests(unittest.TestCase):
    def test_conservative_resource_gate_and_nonfinite_rejection(self):
        self.assertIsNotNone(importlib.util.find_spec('original_finetune'),'original-checkpoint admission required')
        from original_finetune import projection
        p=json.loads((Path(__file__).parent/'original_finetune.json').read_text())
        result=projection(p,4.,2.,2000000,80*1024**3)
        self.assertTrue(result['admitted'])
        self.assertEqual(result['projected_seconds'],(2400*4+108*2)*1.5+240)
        for t,v,m,d in [(5.,2.,2000000,80*1024**3),(4.,2.,5000000,80*1024**3),
                        (4.,2.,2000000,5*1024**3),(float('nan'),2.,2000000,80*1024**3),
                        (0.,2.,2000000,80*1024**3)]:
            self.assertFalse(projection(p,t,v,m,d)['admitted'])

    def test_measurements_are_exclusive_and_do_not_overwrite_prior_steps(self):
        import original_finetune as api
        self.assertTrue(hasattr(api,"save_measurement"))
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            api.save_measurement(folder,[{"seconds":1.}])
            api.save_measurement(folder,[{"seconds":1.},{"seconds":2.}])
            self.assertEqual(len(list(folder.glob("measurement_*.json"))),2)
            with self.assertRaises(FileExistsError):api.save_measurement(folder,[{"seconds":9.}])

    def test_initialization_and_frozen_gate(self):
        p=json.loads((Path(__file__).parent/'original_finetune.json').read_text())
        old=json.loads((Path(__file__).parent/'campaign_segmented_300.json').read_text())
        self.assertEqual(p['gate'],old['gate'])
        self.assertEqual(p['eligible_order'],old['eligible_order'])
        self.assertEqual(p['checkpoint_sha256'],'b2668971dd1230cdbc93bbc7d0182b46c91e7469717dc210052fa4eb20fa8bdb')
        self.assertEqual(p['internal_min_size'],800)
        self.assertEqual(p['internal_max_size'],1333)
        self.assertEqual(p['epochs']*p['training_samples'],2400)
        self.assertFalse(p['train_backbone'])
        self.assertEqual(p['precision'],'float32')


if __name__=='__main__':unittest.main()
