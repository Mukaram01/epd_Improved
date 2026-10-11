"""Resume contract tests use actual deterministic CPU torch/SGD, never real scenes."""
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from campaign import epoch_order

class ContinuationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('continuation'),'resume helpers required')
        import continuation
        return continuation

    def test_actual_torch_uninterrupted_equals_resumed(self):
        api=self.api();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        def initialize():
            random.seed(37);np.random.seed(37);torch.manual_seed(37)
            model=torch.nn.Sequential(torch.nn.Linear(3,7),torch.nn.Dropout(.25),torch.nn.Linear(7,1))
            optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0001)
            return model,optimizer
        def run(model,optimizer,start,end):
            order=[]
            for epoch in range(start,end+1):
                model.train()
                for index in epoch_order(5,37,epoch):
                    x=torch.tensor([[index/5,random.random(),float(np.random.random())]])+torch.rand(1,3)*.01
                    optimizer.zero_grad();loss=model(x).square().sum();loss.backward();optimizer.step()
                    order.append((epoch,index))
                # Validation can consume RNG: checkpoint must follow it.
                model.eval();random.random();np.random.random();torch.rand(3)
            return order
        model,opt=initialize();continuous=run(model,opt,1,8)
        expected=api.capture_state(model,opt,torch,37,8,40,{'fixture':1})
        model,opt=initialize();first=run(model,opt,1,4)
        state=api.capture_state(model,opt,torch,37,4,20,{'fixture':1})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'epoch_04.pt';record=api.atomic_checkpoint(path,state,torch)
            # Construction consumes RNG before exact restoration.
            model,opt=initialize()
            restored=api.load_checkpoint(path,record['sha256'],torch)
            api.restore_state(restored,model,opt,torch)
            rest=run(model,opt,5,8)
        actual=api.capture_state(model,opt,torch,37,8,40,{'fixture':1})
        self.assertEqual(continuous,first+rest)
        def equal(a,b):
            if torch.is_tensor(a):self.assertTrue(torch.equal(a,b))
            elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
            elif isinstance(a,dict):
                self.assertEqual(a.keys(),b.keys())
                for key in a:equal(a[key],b[key])
            elif isinstance(a,(tuple,list)):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):equal(x,y)
            else:self.assertEqual(a,b)
        equal(expected,actual)

    def test_checkpoint_atomic_and_corruption_rejected_before_load(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'epoch_04.pt'
            record=api.atomic_checkpoint(path,{'fixture':torch.ones(2)},torch)
            self.assertEqual(list(Path(tmp).glob('*.partial')),[])
            with self.assertRaises(FileExistsError):api.atomic_checkpoint(path,{},torch)
            path.write_bytes(b'corrupted pickle must never load')
            with self.assertRaisesRegex(ValueError,'hash'):api.load_checkpoint(path,record['sha256'],torch)

    def test_sealed_stage_missing_or_modified_evidence_rejected(self):
        api=self.api()
        from pilot import write_json,sha
        binding={'fixture':'bound'};ids=[f'train_{i}' for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);model=torch.nn.Linear(1,1);opt=torch.optim.SGD(model.parameters(),lr=.1)
            history=[dict(epoch=e,step=(e-1)*3+i+1,scene_id=ids[j])
                for e in range(1,5) for i,j in enumerate(epoch_order(3,37,e))]
            (folder/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
            checkpoints=[]
            for epoch in range(1,5):
                state=api.capture_state(model,opt,torch,37,epoch,epoch*3,binding,config={},campaign={})
                c=api.atomic_checkpoint(folder/f'epoch_{epoch:02d}.pt',state,torch);checkpoints.append(c)
                val=folder/f'validation_{epoch:02d}';val.mkdir();raw=[]
                for i in range(12):
                    path=val/f'scene_{i:03d}.npz';path.write_bytes(b'hash fixture; no image/model prediction')
                    raw.append(dict(path=str(path),sha256=sha(path)))
                metrics=dict(epoch=epoch,raw_predictions=raw)
                write_json(val/'metrics.json',metrics)
                write_json(folder/f'epoch_{epoch:02d}_summary.json',dict(checkpoint=c,metrics=metrics))
            result=dict(status='STAGE_COMPLETE',stage=1,epochs=4,steps=12,binding=binding,checkpoints=checkpoints)
            write_json(folder/'result.json',result);write_json(folder/'process.json',dict(exit_code=0,wall_seconds=1.))
            policy=dict(stages=[[1,4],[5,8]],training_samples=3)
            with self.assertRaises(FileNotFoundError):api.verify_stage(folder,checkpoints[-1]['sha256'],binding,ids,37,torch)
            api.publish_stage(folder,policy)
            api.verify_stage(folder,checkpoints[-1]['sha256'],binding,ids,37,torch)
            path=folder/'epoch_04_summary.json';before=path.read_bytes();path.unlink()
            with self.assertRaisesRegex(ValueError,'missing'):api.verify_stage(folder,checkpoints[-1]['sha256'],binding,ids,37,torch)
            path.write_bytes(before+b' ')
            with self.assertRaisesRegex(ValueError,'modified'):api.verify_stage(folder,checkpoints[-1]['sha256'],binding,ids,37,torch)

    def test_resume_evidence_rejects_drift_and_repeated_steps(self):
        api=self.api()
        binding={'dataset':'datahash','recipe':'recipehash','policy':'policyhash','sources':{'runner':'sourcehash'}}
        history=[dict(epoch=e,step=(e-1)*3+i+1,scene_id=f'train_{j}')
                 for e in range(1,5) for i,j in enumerate(epoch_order(3,37,e))]
        state=dict(epoch=4,step=12,seed=37,binding=binding,smoke_only=False)
        api.validate_progress(state,history,binding,['train_0','train_1','train_2'],37,4)
        import copy
        for key in ('dataset','recipe','policy','sources'):
            bad=copy.deepcopy(binding);bad[key]='changed'
            with self.assertRaises(ValueError):api.validate_progress(state,history,bad,['train_0','train_1','train_2'],37,4)
        for key,value in [('epoch',3),('step',11),('seed',38),('smoke_only',True)]:
            bad=dict(state,**{key:value})
            with self.assertRaises(ValueError):api.validate_progress(bad,history,binding,['train_0','train_1','train_2'],37,4)
        for bad in (history[:-1],history+[history[-1]],history[:1]+history[:1]+history[2:]):
            with self.assertRaises(ValueError):api.validate_progress(state,bad,binding,['train_0','train_1','train_2'],37,4)

if __name__=='__main__':unittest.main()
