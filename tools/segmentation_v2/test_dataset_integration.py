import copy
import unittest
import numpy as np
from expansion import CAMERA
import integrate_dataset


class IntegrationTests(unittest.TestCase):
    def test_geometric_occlusion_is_measurement_only(self):
        obj=dict(size=[.05]*3,pose=[0,0,.04,0,0,0])
        hull=integrate_dataset.projected_silhouette(obj,CAMERA)
        self.assertEqual(hull.dtype,np.dtype('bool'))
        self.assertEqual(hull.shape,(512,512));self.assertGreater(hull.sum(),100)
        ids=np.zeros((512,512),dtype='<u4');ids[hull]=7;before=ids.copy()
        stat=integrate_dataset.occlusion_stat(obj,ids,7,CAMERA)
        self.assertEqual(stat['estimated_occluded_fraction'],0.)
        self.assertTrue(np.array_equal(ids,before))
        ids[hull]=8
        self.assertEqual(integrate_dataset.occlusion_stat(obj,ids,7,CAMERA)['estimated_occluded_fraction'],1.)
    def test_global_membership_rejects_validation_contamination(self):
        from pilot import scene_plan
        original=scene_plan()['scenes']
        from expansion import expansion_plan
        new=expansion_plan()['scenes']
        integrate_dataset.check_membership(original,new)
        bad=copy.deepcopy(new);bad[0]['seed']=original[24]['seed']
        with self.assertRaises(ValueError):integrate_dataset.check_membership(original,bad)
        bad=copy.deepcopy(new);bad[0]['scene_id']=original[36]['scene_id']
        with self.assertRaises(ValueError):integrate_dataset.check_membership(original,bad)
    def test_smoke_rejects_bad_supervised_tensors_without_model(self):
        import torch
        image=torch.zeros(3,512,512,dtype=torch.float32)
        target=dict(boxes=torch.empty(0,4),labels=torch.empty(0,dtype=torch.int64),masks=torch.empty(0,512,512,dtype=torch.uint8))
        integrate_dataset.validate_tensors(image,target,torch)
        with self.assertRaises(ValueError):integrate_dataset.validate_tensors(image.permute(1,2,0),target,torch)
        bad=dict(target,labels=torch.empty(0,dtype=torch.float32))
        with self.assertRaises(ValueError):integrate_dataset.validate_tensors(image,bad,torch)

    def test_batch_seals_require_exact_coverage_and_acquisition_hashes(self):
        import json,tempfile
        from pathlib import Path
        from pilot import sha
        from expansion import expansion_plan
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'attempts').mkdir();(root/'batches').mkdir();plan=expansion_plan()
            for i,spec in enumerate(plan['scenes']):
                (root/'attempts'/f'{i:03d}.json').write_text(json.dumps(dict(scene_id=spec['scene_id'],status='ACQUIRED',sha256={})))
            hashes={p.name:sha(p) for p in sorted((root/'attempts').glob('*.json'))}
            acquisition=dict(status='ACQUIRED',attempt_manifest_sha256=hashes)
            (root/'acquisition.json').write_text(json.dumps(acquisition))
            for batch in range(6):
                (root/'batches'/f'batch_{batch:02d}.json').write_text(json.dumps(dict(batch=batch,status='ACQUIRED',
                    attempt_sha256={f'{i:03d}.json':hashes[f'{i:03d}.json'] for i in range(batch*46,(batch+1)*46)})))
            integrate_dataset.verify_batch_seals(root,plan)
            from expansion import verify_completed
            verify_completed(root,{},plan)
            path=root/'batches/batch_01.json';original=path.read_text();bad=json.loads(original)
            bad['batch']=0;path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):verify_completed(root,{},plan)
            with self.assertRaises(ValueError):integrate_dataset.verify_batch_seals(root,plan)
            bad=json.loads(original);bad['attempt_sha256']={};path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):integrate_dataset.verify_batch_seals(root,plan)
            path.write_text(original);acquisition['attempt_manifest_sha256']={}
            (root/'acquisition.json').write_text(json.dumps(acquisition))
            with self.assertRaises(ValueError):integrate_dataset.verify_batch_seals(root,plan)

if __name__=='__main__':unittest.main()
