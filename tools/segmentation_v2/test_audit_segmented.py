"""Segmented evidence must reject gaps, seal changes and selection drift."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import audit_segmented as audit
from campaign import epoch_order
from pilot import sha


class SegmentedAuditTests(unittest.TestCase):
    def test_history_rejects_missing_repeated_and_wrong_order_steps(self):
        ids=['a','b','c'];seed=42
        rows=[dict(epoch=e,step=(e-1)*3+i+1,scene_id=ids[j])
              for e in range(1,9) for i,j in enumerate(epoch_order(3,seed,e))]
        audit.verify_history(rows[:12],rows[12:],ids,seed)
        bad=copy.deepcopy(rows);bad[12]['step']=12
        with self.assertRaises(ValueError):audit.verify_history(bad[:12],bad[12:],ids,seed)
        with self.assertRaises(ValueError):audit.verify_history(rows[:12],rows[13:],ids,seed)
        bad=copy.deepcopy(rows);bad[13]['scene_id']=bad[12]['scene_id']
        with self.assertRaises(ValueError):audit.verify_history(bad[:12],bad[12:],ids,seed)

    def test_seal_rejects_modified_missing_extra_and_partial_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);p=folder/'proof.json';p.write_text('{}')
            binding={'source':'pinned'}
            seal=dict(schema='epd_v2_completed_stage/1',stage=2,binding=binding,files={'proof.json':sha(p)})
            (folder/'stage_complete.json').write_text(json.dumps(seal))
            audit.verify_seal(folder,2,binding)
            p.write_text('changed')
            with self.assertRaises(ValueError):audit.verify_seal(folder,2,binding)
            p.unlink()
            with self.assertRaises(ValueError):audit.verify_seal(folder,2,binding)
            p.write_text('{}');extra=folder/'extra';extra.touch()
            with self.assertRaises(ValueError):audit.verify_seal(folder,2,binding)
            extra.unlink()
            with self.assertRaises(ValueError):audit.verify_seal(folder,2,{'source':'stale'})
            partial=folder/'checkpoint.partial';partial.touch()
            seal['files']['checkpoint.partial']=sha(partial)
            (folder/'stage_complete.json').write_text(json.dumps(seal))
            with self.assertRaises(ValueError):audit.verify_seal(folder,2,binding)

    def test_selection_never_promotes_rejected_diagnostic(self):
        rows=[dict(epoch=1,recovered=17,mean_iou=.9,background_spill=1,eligible=False,checkpoint={'sha256':'a'}),
              dict(epoch=2,recovered=18,mean_iou=.7,background_spill=10,eligible=True,checkpoint={'sha256':'b'})]
        self.assertEqual(audit.choose(rows)['sha256'],'b')
        self.assertIsNone(audit.choose(rows[:1]))


if __name__=='__main__':unittest.main()
