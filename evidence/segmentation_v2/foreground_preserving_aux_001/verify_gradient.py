"""Synthetic backward verification only; no checkpoint, inference or SGD step."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools/segmentation_v2'))
import torch
import cv2
from test_native_boundary_aux import synthetic_model,target
from foreground_preserving_aux import training_losses,non_mask_snapshot,assert_frozen,MASK_PREFIXES
from native_boundary_candidate import state_hash
from diagnostic_shadow_capture import durable_json
from shadow_mask_paste import digest

torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(2)
torch.use_deterministic_algorithms(True)
model=synthetic_model();before=non_mask_snapshot(model);all_before={n:t.clone() for n,t in model.state_dict().items()}
info={};parts=training_losses(model,[torch.ones(3,512,512)*.4],[target()],.25,diagnostics=info)
parts['loss_mask_native_boundary'].backward();assert_frozen(model,before)
norms={n:float(p.grad.abs().sum()) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
assert len(norms)==12 and all(v>0 for v in norms.values())
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith(MASK_PREFIXES))
assert all(torch.equal(t,model.state_dict()[n]) for n,t in all_before.items())
durable_json(Path(sys.argv[1]),{'scope':'Synthetic4channel model/64internal raster; native512GT; actualROIHeads/TVpaste; backward only',
    'weighted_auxiliary_loss':float(parts['loss_mask_native_boundary'].detach()),'diagnostics':info,
    'parameter_gradient_l1':norms,'mask_tensors_with_finite_nonzero_gradient':len(norms),
    'non_mask_state_before_sha256':state_hash(before),'non_mask_state_after_sha256':state_hash(non_mask_snapshot(model)),
    'all_state_unchanged_no_optimizer_step':True,'real_training_steps':0,'trained_checkpoint_loaded':False,
    'source_sha256':digest(Path(__file__).resolve()),'stage_a2':'BLOCKED_0/2','original_training_supervision':'UNKNOWN'})
