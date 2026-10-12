# Segmented 300-scene CPU campaign

The versioned `campaign_segmented_300.json` policy executes exactly eight epochs
as Stage1 (epochs1–4) and Stage2 (epochs5–8), each with300 samples/epoch andbatch1.
Each process is bounded by10800s and6144MiB RSS; Stage2 is additionally bounded
by the remaining18000s aggregate execution allowance. The5GiB diskreserve,
seed260110, architecture, SGD settings, confidence>=.8, masks>.5 and validation
gate/ranking are unchanged. The failed full-run10800s preflight remains failed.

`measure_segmented.py` measures Stage1 without optimizer updates. The projection
includes1200 measured worst-case forward/backward steps and48 validation forwards
perstage, with the existing1.5 factor and120s overhead each. Both stage andaggregate
projections must pass. This is a new segmented policy, not a replacement historical
preflight. Resource measurement itself is separately bounded at180s/6144MiB.
Aggregate campaign accounting charges bounded worker wall time, including startup,
hash checks, tensor preparation, validation, checkpoint IO and final report creation.
Preflight and post-run verification wall time are recorded separately.

## Exact epoch-boundary continuation

- Each segmented checkpoint is written after validation, immediately before the
  next epoch boundary. It contains model, complete SGD/momentum state, Python,
  NumPy andCPU Torch RNG states, seed, epoch, global step, recipe, policy andbindings.
- Save writes an exclusive `.partial`, flushes/fsyncs, atomically hardlinks the
  final checkpoint without replacement, removes `.partial` andfsyncs thedirectory.
- Stage directories are exclusive. A timeout/failure preserves partial evidence
  and cannot publish `stage_complete.json`. No automatic retry or repair occurs.
- Successful completion seals all stage files, including checkpoints, history,
  raw validation outputs, metrics, executed-source snapshots andnative process
  result. Continuation requires the independently supplied epoch4 SHA256 plus
  matching source/recipe/policy/dataset/dependency identities andall sealedhashes.
- Verification requires exactly steps1..1200, every frozen training sample once
  per epoch in the same seeded order, checkpointepochs1..4, complete summaries,
  all12 raw validation outputs per epoch andsuccessful native exit. Missing,
  modified, extra, duplicated or incomplete evidence fails closed.
- Model andoptimizer construction happens before restoring state. Restore strictly
  loads parameters andSGD state andsets all RNG states. Epoch order is generated
  by isolated `random.Random(seed+epoch-1)`, exactly as uninterrupted execution.
  There is no scheduler or implicit learning-rate reset; saved optimizer groups
  and momentum are restored. Stage2 continues atstep1201/epoch5 andselects among
  all eight epochs, including retained Stage1 eligibility evidence.

The deterministic actual-PyTorch fixture uses stochastic dropout, Python/NumPy/
Torch random draws, SGD momentum andweight decay, andRNG-consuming validation.
Continuous8 and4+4 produce bit-identical model parameters, optimizer tensors,
RNG states, sample orders and40 total fixture steps. This proves the implemented
continuation state contract on the installed deterministic CPU framework; itdoes
not claim an extra uninterrupted Mask R-CNN training campaign was run.

## Prepare, then resume in a later task

After Stage1, `prepare_stage2.py` verifies the completed stage andindependently
loads epoch4, restores its actual model/optimizer/RNG state without training or
inference, andwrites Stage2 feasibility/identity files. Remaining projection uses
1200 * maximum observed full training-step wall plus4 * maximum observed complete
validation-epoch wall, with unchanged1.5 factor+120s. Admission additionally requires
remaining aggregate time, memory anddisk checks. Stage2 is not started by preparation.

```bash
python3 tools/segmentation_v2/prepare_stage2.py \
  --stage1 /external/campaign/stage1 --checkpoint-sha256 <verified-epoch4-sha> \
  --artifacts /external/composite/dataset.json --weights /external/pinned-coco.pth \
  --output /external/campaign/stage2-preflight
# A later explicitly authorized task only, with feasible=true:
python3 tools/segmentation_v2/campaign.py train \
  --policy tools/segmentation_v2/campaign_segmented_300.json --stage 2 \
  --resume-stage /external/campaign/stage1 --resume-sha256 <verified-epoch4-sha> \
  --artifacts /external/composite/dataset.json --weights /external/pinned-coco.pth \
  --preflight /external/campaign/stage2-preflight --output /external/campaign/stage2
```

Use current hash-bound sources/dependencies unchanged when resuming. A source,
dataset orconfiguration change requires investigation, not bypassing thebinding.
No held-out/historical regression inference, export, deployment, production adapter,
association, EMD orphysical safety changes are part of this workflow. Renderer IDs
remain offline supervision/scoring only; the model receives RGB tensors exclusively.
