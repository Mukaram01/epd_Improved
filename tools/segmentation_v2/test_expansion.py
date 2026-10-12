"""Arithmetic fixtures are tests, never genuine dataset evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import expansion


class ExpansionTests(unittest.TestCase):
    def test_frozen_plan_variety_and_no_pilot_collision(self):
        plan=expansion.expansion_plan()
        self.assertEqual(plan,expansion.expansion_plan())
        self.assertEqual(len(plan['scenes']),276)
        self.assertEqual(sum(s['negative'] for s in plan['scenes']),46)
        self.assertEqual({s['split'] for s in plan['scenes']},{'train'})
        expansion.validate_plan(plan)
        for field in ('scene_id','seed'):
            bad=copy.deepcopy(plan);bad['scenes'][1][field]=bad['scenes'][0][field]
            with self.assertRaises(ValueError):expansion.validate_plan(bad)
        bad=copy.deepcopy(plan);bad['scenes'][1]['split']='validation'
        with self.assertRaises(ValueError):expansion.validate_plan(bad)
        self.assertEqual({s['arrangement'] for s in plan['scenes'] if not s['negative']},
                         {'separated','touching','partial_occlusion'})
    def test_frozen_manifest_exclusive_and_contract_change_rejected(self):
        from pilot import write_json
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scenes.json';plan=expansion.expansion_plan();write_json(path,plan)
            with self.assertRaises(FileExistsError):write_json(path,plan)
            bad=copy.deepcopy(plan);bad['camera']['position'][0]=.01
            with self.assertRaises(ValueError):expansion.validate_plan(bad)

    def test_identity_collision_ownership_and_negative_labels(self):
        spec={'negative':False,'objects':[dict(name='cube',class_id=1,size=[.05]*3)]}
        report={'mapping':[dict(model='cube',class_id=1,renderer_id=7,item_id=0,
                               visual_entity=12,link_entity=11,model_entity=10)],
                'inventory':{'complete':True,'models':[dict(id=10,parent=1)],
                             'links':[dict(id=11,parent=10)],
                             'visuals':[dict(id=12,parent=11,geometry=dict(type='BOX',size=[.05]*3))],
                             'collisions':[dict(id=13,parent=11,geometry=dict(type='BOX',size=[.05]*3))]}}
        ids=np.zeros((512,512),dtype='<u4');ids[1:4,2:5]=7
        stats=expansion.label_integrity(spec,report,ids)
        self.assertEqual(stats['visible_objects'],1)
        self.assertEqual(stats['instances'][0]['visible_pixels'],9)
        bad=copy.deepcopy(report);bad['inventory']['collisions'][0]['parent']=99
        with self.assertRaises(ValueError):expansion.label_integrity(spec,bad,ids)
        with self.assertRaises(ValueError):expansion.label_integrity(dict(spec,negative=True),report,ids)
        wrong=ids.copy();wrong[0,0]=99
        with self.assertRaises(ValueError):expansion.label_integrity(spec,report,wrong)
        with self.assertRaises(ValueError):expansion.label_integrity(spec,report,ids.astype('f4'))
    def test_duplicate_rgb_rejected_against_other_splits_and_historical(self):
        seen={'held-out-hash'}
        with self.assertRaises(ValueError):expansion.register_rgb('held-out-hash',seen)
        expansion.register_rgb('new-hash',seen)
        with self.assertRaises(ValueError):expansion.register_rgb('new-hash',seen)
        from pilot import HISTORICAL_RGB
        with self.assertRaises(ValueError):expansion.register_rgb(next(iter(HISTORICAL_RGB)),seen)
    def test_partial_campaign_is_not_complete_and_attempts_are_never_reused(self):
        plan=expansion.expansion_plan()
        attempted=[dict(scene_id=plan['scenes'][0]['scene_id'],status='ACQUIRED'),
                   dict(scene_id=plan['scenes'][1]['scene_id'],status='FAILED')]
        state=expansion.progress_state(plan,attempted)
        self.assertEqual(state['successful'],1);self.assertEqual(len(state['unattempted']),274)
        self.assertFalse(state['complete']);self.assertEqual(state['failed'],[attempted[1]['scene_id']])
        with self.assertRaises(ValueError):expansion.progress_state(plan,attempted+[attempted[0]])
        with self.assertRaises(ValueError):expansion.require_complete(plan,attempted)
    def test_loader_filters_membership_before_pixel_access_and_legacy_delegates(self):
        from unittest.mock import patch
        import dataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'dataset.json'
            from pilot import scene_plan,sha
            plan=scene_plan()['scenes'];new=expansion.expansion_plan()['scenes']
            pilot=root/'pilot';extra=root/'extra';pilot.mkdir();extra.mkdir();(extra/'attempts').mkdir()
            def record(spec):
                return dict(scene_id=spec['scene_id'],seed=spec['seed'],split=spec['split'],status='ACQUIRED',
                            sha256={'capture.json.rgb8':spec['scene_id']+'-rgb'})
            original=[record(spec) for spec in plan]
            (pilot/'acquisition.json').write_text(json.dumps(dict(status='ACQUIRED',scenes=original)))
            for i,spec in enumerate(new):(extra/'attempts'/f'{i:03d}.json').write_text(json.dumps(record(spec)))
            hashes={p.name:sha(p) for p in (extra/'attempts').glob('*.json')}
            (extra/'acquisition.json').write_text(json.dumps(dict(status='ACQUIRED',attempt_manifest_sha256=hashes)))
            records=[dict(record(spec),folder=str((pilot if spec in plan else extra)/'dataset'/spec['scene_id']),
                          origin='pilot_001' if spec in plan else 'expansion_002') for spec in plan+new]
            data=dict(schema='epd_v2_dataset/2',records=records,pilot_root=str(pilot),expansion_root=str(extra),
                pilot_manifest_sha256={'acquisition.json':sha(pilot/'acquisition.json')},
                expansion_acquisition_sha256=sha(extra/'acquisition.json'))
            manifest.write_text(json.dumps(data))
            self.assertEqual(len(dataset.enumerate_records(manifest,'train')),300)
            import train
            with patch('dataset.verified_rows',return_value=iter(['integrated'])):
                self.assertEqual(list(train.verified_rows(manifest,'train')),['integrated'])
            for mutation in ('split_swap','folder_substitution','hash_substitution'):
                bad=copy.deepcopy(data)
                if mutation=='split_swap':
                    bad['records'][0]['split']='validation';bad['records'][24]['split']='train'
                elif mutation=='folder_substitution':bad['records'][0]['folder']=bad['records'][24]['folder']
                else:bad['records'][0]['sha256']=bad['records'][24]['sha256']
                manifest.write_text(json.dumps(bad))
                with self.assertRaises(ValueError,msg=mutation):dataset.enumerate_records(manifest,'train')
            bad=copy.deepcopy(records);bad[1]=bad[0]
            manifest.write_text(json.dumps(dict(schema='epd_v2_dataset/2',records=bad)))
            with self.assertRaises(ValueError):dataset.enumerate_records(manifest,'train')
            with patch('dataset.pilot_rows',return_value=iter(['legacy'])) as legacy:
                self.assertEqual(list(dataset.verified_rows(root,'train')),['legacy'])
                legacy.assert_called_once_with(root,'train')

if __name__=='__main__':unittest.main()
