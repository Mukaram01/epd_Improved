import importlib.util
import unittest


class MaskFinetuneTests(unittest.TestCase):
    def test_weight_only_mask_loss_without_changing_other_losses(self):
        self.assertIsNotNone(importlib.util.find_spec('mask_finetune'))
        from mask_finetune import weighted_loss
        self.assertEqual(weighted_loss({'loss_mask':3.,'loss_classifier':2.,'loss_box_reg':1.},2.),9.)
        with self.assertRaises(ValueError):weighted_loss({'loss_classifier':2.},2.)

    def test_frozen_gate_and_conservative_admission(self):
        from mask_finetune import policy,admission
        from campaign import eligible
        p=policy()
        self.assertEqual((p['epochs'],p['training_samples'],p['mask_loss_weight']),(2,300,2.))
        a=admission(p)
        self.assertTrue(a['admitted'])
        self.assertEqual(a['optimizer_steps'],600)
        self.assertLess(a['projected_seconds'],10800)
        m=dict(positives=22,recovered=18,mean_iou=.7,false_positives=1,background_spill=1914,contract_violations=0)
        self.assertTrue(eligible(m,p));m['background_spill']=1915
        self.assertFalse(eligible(m,p))


if __name__=='__main__':unittest.main()
