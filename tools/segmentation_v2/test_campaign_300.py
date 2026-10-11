"""Training integration: a truncated or drifting campaign must fail these tests."""
import json
from pathlib import Path
import unittest
import campaign

class ExpandedCampaignTests(unittest.TestCase):
    def test_order_visits_all_300_once_per_epoch(self):
        self.assertTrue(hasattr(campaign,'epoch_order'), 'campaign needs reusable deterministic epoch ordering')
        orders=[campaign.epoch_order(300,260110,e) for e in range(1,9)]
        self.assertEqual(sum(map(len,orders)),2400)
        for e,order in enumerate(orders,1):
            self.assertEqual(sorted(order),list(range(300)))
            self.assertEqual(order,campaign.epoch_order(300,260110,e))
        self.assertNotEqual(orders[0],orders[1])

    def test_expansion_policy_preserves_recipe_gate_and_legacy(self):
        path=Path(__file__).parent/'campaign_300.json'
        self.assertTrue(path.exists(),'explicit expanded policy required')
        new=json.loads(path.read_text());old=json.loads((path.parent/'campaign.json').read_text())
        for field in ('gate','eligible_order','epochs','batch_size','seed'):
            self.assertEqual(new[field],old[field])
        self.assertEqual(new['training_samples'],300)
        self.assertEqual(new['cpu_timeout_seconds'],10800)
        self.assertEqual(old['cpu_timeout_seconds'],3600)

    def test_training_loader_enforces_counts_and_split_isolation(self):
        self.assertTrue(hasattr(campaign,'campaign_rows'),'worker needs validated split integration')
        from unittest.mock import patch
        rows={s:[({'scene_id':f'{s}_{i}'},None,None,None) for i in range(n)]
              for s,n in [('train',300),('validation',12)]}
        accessed=[]
        def loader(source,split):
            accessed.append(split)
            return iter(rows[split])
        with patch('campaign.verified_rows',side_effect=loader):
            train,val=campaign.campaign_rows(Path('metadata.json'),{'training_samples':300})
            self.assertEqual(len(train),300);self.assertEqual(len(val),12)
            self.assertEqual(accessed,['train','validation'])
            rows['train'].pop()
            with self.assertRaisesRegex(ValueError,'sample counts'):
                campaign.campaign_rows(Path('metadata.json'),{'training_samples':300})
            rows['train'].append(rows['validation'][0])
            with self.assertRaisesRegex(ValueError,'split overlap'):
                campaign.campaign_rows(Path('metadata.json'),{'training_samples':300})

    def test_both_devices_require_matching_successful_feasibility(self):
        self.assertTrue(hasattr(campaign,'require_feasibility'),'CUDA must not bypass feasibility')
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);policy={'training_samples':300}
            for device in ('cpu','cuda'):
                path=root/'feasibility.json'
                path.write_text(json.dumps(dict(device=device,feasible=False,optimizer_updates=0)))
                with self.assertRaises(ValueError):campaign.require_feasibility(root,device,policy)
                path.write_text(json.dumps(dict(device=device,feasible=True,optimizer_updates=0)))
                campaign.require_feasibility(root,device,policy)
                other='cuda' if device=='cpu' else 'cpu'
                with self.assertRaises(ValueError):campaign.require_feasibility(root,other,policy)
                path.write_text(json.dumps(dict(device=device,feasible=True,optimizer_updates=1)))
                with self.assertRaises(ValueError):campaign.require_feasibility(root,device,policy)

    def test_composite_identity_is_hash_bound(self):
        self.assertTrue(hasattr(campaign,'dataset_identity'),'support manifest file identity')
        import tempfile
        from pilot import sha
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'dataset.json';source.write_text('{}')
            self.assertEqual(campaign.dataset_identity(source),sha(source))
            folder=Path(tmp)/'pilot';folder.mkdir();(folder/'acquisition.json').write_text('{}')
            self.assertEqual(campaign.dataset_identity(folder),sha(folder/'acquisition.json'))

if __name__=='__main__':unittest.main()
