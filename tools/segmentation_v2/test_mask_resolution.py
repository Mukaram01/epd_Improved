import unittest
import torch
from recover_original import make_model


class MaskResolutionTests(unittest.TestCase):
    def test_real_pool_and_predictor_double_logits_without_replacing_weights(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('mask_resolution'))
        from mask_resolution import configure56
        torch.set_num_threads(2)
        model = make_model()
        before = {k: v.data_ptr() for k, v in model.state_dict().items()}
        self.assertEqual(model.roi_heads.mask_roi_pool.output_size, (14, 14))
        configure56(model)
        self.assertEqual(before, {k: v.data_ptr() for k, v in model.state_dict().items()})
        features = {str(i): torch.rand(1, 256, 64 // 2**i, 64 // 2**i) for i in range(4)}
        for boxes, count in [(torch.tensor([[4., 5., 50., 55.]]), 1), (torch.empty(0, 4), 0)]:
            pooled = model.roi_heads.mask_roi_pool(features, [boxes], [(256, 256)])
            logits = model.roi_heads.mask_predictor(model.roi_heads.mask_head(pooled))
            self.assertEqual(tuple(logits.shape), (count, 2, 56, 56))
        self.assertTrue(all(not v.requires_grad for n, v in model.named_parameters()
                            if not n.startswith(('roi_heads.mask_head.', 'roi_heads.mask_predictor.'))))


if __name__ == '__main__':
    unittest.main()
