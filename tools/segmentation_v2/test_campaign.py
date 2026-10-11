import copy
import json
from pathlib import Path
import unittest
import numpy as np
from campaign import eligible, checkpoint_key, validate_prediction, score_predictions

class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.policy=json.loads((Path(__file__).parent/'campaign.json').read_text())
        self.good=dict(positives=22,recovered=18,mean_iou=.7,false_positives=1,
                       background_spill=1914,contract_violations=0,epoch=2)
    def test_gate_strict_and_frozen(self):
        self.assertTrue(eligible(self.good,self.policy))
        for key,value in [('recovered',17),('mean_iou',.685890),('false_positives',2),
                          ('background_spill',1915),('contract_violations',1),('positives',21)]:
            bad=dict(self.good,**{key:value});self.assertFalse(eligible(bad,self.policy),key)
    def test_selection_eligible_only_order(self):
        self.assertGreater(checkpoint_key(dict(self.good,recovered=19)),checkpoint_key(self.good))
        self.assertGreater(checkpoint_key(dict(self.good,epoch=1)),checkpoint_key(self.good))
    def test_output_contract_empty_and_invalid(self):
        empty=dict(boxes=np.empty((0,4),np.float32),labels=np.empty(0,np.int64),
                   scores=np.empty(0,np.float32),masks=np.empty((0,1,4,4),np.float32))
        validate_prediction(empty,4)
        bad=dict(empty,labels=np.empty(0,np.float32))
        with self.assertRaises(ValueError):validate_prediction(bad,4)
        one=dict(boxes=np.array([[0,0,4,4]],np.float32),labels=np.array([1],np.int64),
                 scores=np.array([.8],np.float32),masks=np.ones((1,1,4,4),np.float32))
        validate_prediction(one,4)
        for field in ('boxes','scores','masks'):
            bad=copy.deepcopy(one);bad[field].flat[0]=np.nan
            with self.assertRaises(ValueError):validate_prediction(bad,4)
    def test_scoring_negative_false_positive_and_threshold(self):
        ids=np.zeros((4,4),np.uint32);ids[1:3,1:3]=7
        masks=np.zeros((1,1,4,4),np.float32);masks[0,0,1:3,1:3]=.5001
        pred=dict(boxes=np.array([[1,1,3,3]],np.float32),labels=np.array([1],np.int64),
                  scores=np.array([.8],np.float32),masks=masks)
        metrics=score_predictions('scene_025',pred,ids,{7})
        self.assertEqual(metrics['recovered'],1);self.assertEqual(metrics['mean_iou'],1)
        self.assertEqual(metrics['background_spill'],0)
        neg=score_predictions('scene_024',pred,np.zeros_like(ids),set())
        self.assertEqual(neg['false_positives'],1);self.assertEqual(neg['background_spill'],4)
        pred['masks'][0,0,1:3,1:3]=.5
        miss=score_predictions('scene_025',pred,ids,{7})
        self.assertEqual(miss['recovered'],0);self.assertEqual(miss['missing_visible'],4)
        self.assertEqual(miss['instances'][0]['renderer_identity'],7)


class IdentityAuditTests(unittest.TestCase):
    def test_mapping_class_and_unique_ownership(self):
        from campaign import validate_identity_mapping
        spec={'objects':[{'name':'cube_0','class_id':1,'size':[.1,.1,.1]}]}
        report={'mapping':[dict(model='cube_0',class_id=1,renderer_id=7,item_id=0,
                               visual_entity=12,link_entity=11,model_entity=10)],
                'inventory':{'complete':True,'collisions':[],
                             'models':[{'id':10,'parent':1}],
                             'links':[{'id':11,'parent':10}],
                             'visuals':[{'id':12,'parent':11,'geometry':{'type':'BOX','size':[.1,.1,.1]}}]}}
        validate_identity_mapping(spec,report)
        bad=copy.deepcopy(report);bad['mapping'][0]['class_id']=0
        with self.assertRaises(ValueError):validate_identity_mapping(spec,bad)
        bad=copy.deepcopy(report);bad['mapping'].append(bad['mapping'][0].copy())
        with self.assertRaises(ValueError):validate_identity_mapping(spec,bad)
        bad=copy.deepcopy(report);bad['mapping'][0]['link_entity']=99
        with self.assertRaises(ValueError):validate_identity_mapping(spec,bad)

class OfflineAuditTests(unittest.TestCase):
    def test_replay_and_checkpoint_identity(self):
        import tempfile
        from unittest.mock import patch
        from audit_campaign import audit
        from pilot import sha
        from campaign import score_predictions
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);train=root/'training';train.mkdir();folder=train/'validation_01';folder.mkdir()
            ids=np.zeros((4,4),np.uint32);ids[1:3,1:3]=7
            masks=np.zeros((1,1,4,4),np.float32);masks[0,0,1:3,1:3]=1
            pred=dict(boxes=np.array([[1,1,3,3]],np.float32),labels=np.array([1],np.int64),
                      scores=np.array([.8],np.float32),masks=masks)
            path=folder/'scene_025.npz';np.savez(path,**pred)
            metrics=score_predictions('scene_025',pred,ids,{7})
            metrics['mean_boundary_px']=0.
            metrics['eligible']=False
            metrics['raw_predictions']=[dict(scene_id='scene_025',path=str(path),sha256=sha(path))]
            (folder/'metrics.json').write_text(json.dumps(metrics))
            checkpoint=train/'epoch_01.pt';checkpoint.write_bytes(b'hash-only checkpoint fixture; never deserialized')
            policy=json.loads((Path(__file__).parent/'campaign.json').read_text());policy['epochs']=1
            (train/'run_config.json').write_text(json.dumps(dict(policy=policy,sources={'campaign.json':'fixture'})))
            (train/'result.json').write_text(json.dumps(dict(epochs=1,steps=24,selected=None,
                checkpoints=[dict(epoch=1,path=str(checkpoint),sha256=sha(checkpoint))])))
            history=[dict(epoch=1,step=i+1,scene_id=f'scene_{i:03d}') for i in range(24)]
            (train/'history.jsonl').write_text('\n'.join(json.dumps(r) for r in history)+'\n')
            def rows(_,split):
                if split=='train':return [(dict(scene_id=r['scene_id']),None,None,None) for r in history]
                return [(dict(scene_id='scene_025'),Path('scene_025'),{'mapping':[dict(renderer_id=7,class_id=1)]},ids)]
            with patch('audit_campaign.audit_identities',return_value={'contract_violations':0}), \
                 patch('audit_campaign.verified_rows',side_effect=rows):
                result=audit(root,train)
                self.assertTrue(result['original_metrics_reproduced_exactly'])
                self.assertEqual(result['epochs'][0]['max_boundary_px'],0.)
                self.assertEqual(result['status'],'V2_TRAINED_BUT_REJECTED')
                result_path=train/'result.json';original_result=result_path.read_text()
                incomplete=json.loads(original_result);incomplete['checkpoints']=[]
                result_path.write_text(json.dumps(incomplete))
                with self.assertRaisesRegex(ValueError,'checkpoint epochs'):audit(root,train)
                result_path.write_text(original_result)
                history_path=train/'history.jsonl';original_history=history_path.read_text()
                history_path.write_text(original_history+json.dumps(dict(epoch=2,step=25,scene_id='extra'))+'\n')
                with self.assertRaisesRegex(ValueError,'training history'):audit(root,train)
                history_path.write_text(original_history.replace('"step": 24','"step": 23'))
                with self.assertRaisesRegex(ValueError,'training history'):audit(root,train)
                history_path.write_text(original_history)
                checkpoint.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'checkpoint hash'):audit(root,train)

if __name__=='__main__':unittest.main()
