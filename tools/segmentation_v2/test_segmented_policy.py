import importlib.util
import json
from pathlib import Path
import unittest

class SegmentedPolicyTests(unittest.TestCase):
    def test_projection_retains_safety_and_requires_both_budgets(self):
        self.assertIsNotNone(importlib.util.find_spec('segmented_policy'),'bounded segmented policy required')
        from segmented_policy import project,remaining_admission
        p=json.loads((Path(__file__).parent/'campaign_segmented_300.json').read_text())
        old=json.loads((Path(__file__).parent/'campaign_300.json').read_text())
        for key in ('gate','eligible_order','seed','epochs','batch_size','disk_reserve_gib','feasibility_safety_factor'):
            self.assertEqual(p[key],old[key])
        result=project(p,4.,5.,1700000,80*1024**3)
        self.assertEqual(result['estimated_stage_seconds'],7680.)
        self.assertEqual(result['estimated_aggregate_seconds'],15360.)
        self.assertTrue(result['feasible'])
        self.assertFalse(project(p,5.,5.,1700000,80*1024**3)['feasible'])
        self.assertFalse(project(p,4.,5.,6000000,80*1024**3)['feasible'])
        self.assertFalse(project(p,4.,5.,1700000,5*1024**3)['feasible'])
        admission=remaining_admission(p,[{'seconds':4.}]*1200,[60.]*4,5000.)
        self.assertTrue(admission['admitted'])
        self.assertFalse(remaining_admission(p,[{'seconds':4.}]*1200,[60.]*4,11000.)['admitted'])

if __name__=='__main__':unittest.main()
