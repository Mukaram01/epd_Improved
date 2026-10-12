import importlib.util
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from pilot import sha


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('recover_original'),'safe recovery utility required')
        import recover_original
        self.api=recover_original

    def test_original_frozenbn_uses_coco_v1_zero_epsilon(self):
        from torchvision.ops.misc import FrozenBatchNorm2d
        self.assertTrue(hasattr(self.api,'make_model'),'explicit original norm construction required')
        model=self.api.make_model()
        norms=[m for m in model.modules() if isinstance(m,FrozenBatchNorm2d)]
        self.assertEqual(len(norms),53)
        self.assertTrue(all(m.eps==0 for m in norms))
        self.assertEqual(model.transform.min_size,(800,))
        self.assertEqual(model.transform.max_size,1333)
        self.assertEqual(model.roi_heads.score_thresh,.05)

    def test_safe_loader_rejects_hash_before_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'checkpoint.pth';p.write_bytes(b'not pickle')
            with self.assertRaisesRegex(ValueError,'hash'):self.api.safe_load(p,'0'*64)

    def test_safe_loader_has_no_unsafe_pickle_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'unsafe.pth';torch.save({'custom':UnsafePayload()},p)
            with self.assertRaises(Exception):self.api.safe_load(p,sha(p))

    def test_schema_and_class_heads_fail_closed(self):
        with self.assertRaises(ValueError):self.api.unpack({'state_dict':{}})
        state={'roi_heads.box_predictor.cls_score.weight':torch.zeros(2,1024),
               'roi_heads.box_predictor.bbox_pred.weight':torch.zeros(8,1024),
               'roi_heads.mask_predictor.mask_fcn_logits.weight':torch.zeros(2,256,1,1)}
        c=dict(model_state=state,num_classes=2,class_names=['background','cube'],architecture='maskrcnn_resnet50_fpn')
        self.assertEqual(self.api.unpack(c)[0],state)
        c['num_classes']=3
        with self.assertRaises(ValueError):self.api.unpack(c)

    def test_parity_requires_exact_labels_counts_and_foreground(self):
        p=dict(boxes=np.array([[1,2,3,4]],np.float32),labels=np.array([1],np.int64),scores=np.array([.8],np.float32),masks=np.full((1,1,2,2),.5,np.float32))
        self.assertTrue(self.api.compare(p,p)['passed'])
        q={k:v.copy() for k,v in p.items()};q['masks'][0,0,0,0]=.500001
        r=self.api.compare(p,q);self.assertFalse(r['passed']);self.assertEqual(r['foreground_decision_differences'],1)
        q={k:v.copy() for k,v in p.items()};q['boxes'][0,0]+=.01
        self.assertFalse(self.api.compare(p,q)['passed'])
        q={k:v[:0].copy() for k,v in p.items()}
        self.assertFalse(self.api.compare(p,q)['passed'])
        q={k:v.copy() for k,v in p.items()};q['scores'][0]=np.nan
        self.assertFalse(self.api.compare(p,q)['passed'])

    def test_retained_reference_preserves_confident_masks_without_correction(self):
        self.assertTrue(hasattr(self.api,'confident'),'explicit reference-format handling required')
        p=dict(boxes=np.zeros((2,4),np.float32),labels=np.ones(2,np.int64),scores=np.array([.79,.8],np.float32),
               masks_confident=np.full((1,1,2,2),.501,np.float32))
        r=self.api.confident(p)
        self.assertEqual(r['boxes'].shape,(1,4))
        self.assertEqual(r['masks'].tobytes(),p['masks_confident'].tobytes())
        p['masks_confident']=p['masks_confident'][:0]
        with self.assertRaises(ValueError):self.api.confident(p)

    def test_failed_verification_exits_nonzero_for_automation(self):
        self.assertTrue(hasattr(self.api,'verification_exit_code'),'fail-closed CLI status required')
        self.assertEqual(self.api.verification_exit_code({'classification':'ORIGINAL_MODEL_RECOVERED'}),0)
        self.assertNotEqual(self.api.verification_exit_code({'classification':'BLOCKED — parity mismatch'}),0)
        self.assertNotEqual(self.api.verification_exit_code({}),0)


class UnsafePayload:
    def __reduce__(self):return (eval,('1+1',))


if __name__=='__main__':unittest.main()
