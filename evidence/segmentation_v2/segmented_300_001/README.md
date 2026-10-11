# Stage1 complete; deterministic CPU continuation ready

**V2_STAGE1_COMPLETE_RESUME_READY**

Baseline `b6dbacc61baf066b1a20652176eec3f0b4dcd278`; existing draft PR219.
This is a completed first four-epoch stage, not a qualified model. All four
checkpoints failed the unchanged validation gate. Stage2 was **not executed**.

## Actual execution

- 300 genuine training scenes/epoch,484 visible instances and50 negatives;
  exactly4 epochs and1200 optimizer steps, each training sample once per epoch.
- Fresh SHA-pinned COCO_V1, seed260110, batch1, Torch CPU threads2, deterministic
  algorithms; unchanged frozen-backbone Mask R-CNN/SGD recipe. No rejected model
  used as initialization, no optimizer/scheduler reset or hyperparameter search.
- Stage1 native exit0; bounded worker wall **2697.432626s (44m57s)**.
  Peak RSS **3142616KiB (2.997GiB)**, below6144MiB. The process hit no resource limit.
- Preflight was separately bounded180s/6144MiB and took33.560251s withzero updates.
  New measured projection:7851.668067s/stage and15703.336133s aggregate,
  below10800s/stage and18000s aggregate. The1.5 factor+120s overhead are unchanged.
- Original failed full-run preflight is still false and its hashes are unchanged;
  see[historical_resource_stop_unchanged.json](historical_resource_stop_unchanged.json).

## Frozen validation results

Confidence>=0.8; foreground strictly>0.5. Recall counts are out of22 visible
instances in the original12 validation scenes. Misses contribute zero IoU.

| Epoch | Mean total loss | Recall | Mean IoU | FP | Spill pixels | Cross-object pixels | Missing pixels | Mean boundary px | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|0.624834|9/22|0.376874|8|51125|1372|24806|258.021066|No|
|2|0.437894|9/22|0.363739|4|607|1186|26716|257.796601|No|
|3|0.372394|10/22|0.434413|1|634|226|23465|256.919430|No|
|4|0.343069|11/22|0.435827|1|574|2687|27909|256.955429|No|

Required: recall>=18/22, IoU>0.685890, FP<=1, spill<1915, zero scoped
class/output/identity violations. All epochs hadzero scoped violations andzero
confident detections on both negative validation scenes. Epochs1–4 all fail recall
andIoU; epoch1 also failsFP/spill, epoch2 also failsFP. No gate was relaxed.
Boundary misses retain the existing512px convention. Cross-object contamination
remains reported; no masks were corrected, cropped to IDs, eroded or fitted.

[epoch_table.json](epoch_table.json) retains all loss components andgate failures;
`epoch_01_summary.json` through`epoch_04_summary.json` retain every instance,
detection andnegative-image measurement, raw probability prediction hashes and
checkpoint references. [validation_replay.json](validation_replay.json) proves
all4 metrics andper-instance results recompute exactly from the retained genuine
raw predictions, withno additional model inference.

## Deterministic resume verification

34 focused tests pass, including actual PyTorch stochastic/dropout SGD fixture:
continuous8 epochs versus4+4 produce bit-identical parameters, optimizer tensors,
Python/NumPy/Torch RNG state, sample order andstep counts. Negative tests cover
corrupt checkpoint bytes, wrong epoch/step/seed, duplicate/missing history,
incompatible dataset/recipe/policy/source andmissing/modified stage evidence.
See[tests_final.log](tests_final.log). Compileall andgit diff --check also pass.
This is a small framework fixture, not another real Mask R-CNN campaign.

Actual Stage1 verification then checked every sealed file, source/config/dataset/
dependency binding, all1200 steps in deterministic order, all4 checkpoints andall
48 raw validation outputs. `prepare_stage2.py` loaded epoch4 andstrictly restored
its real model parameters, complete optimizer state/groups andPython/NumPy/Torch
RNG state without inference oroptimizer updates. Verification exit0 in13.685437s.

Epoch4 checkpoint:

```
/home/user/epd-v2-artifacts/segmented-300-001/stage1/epoch_04.pt
SHA256 98bb7afde01d0c013df0c4e40ba8cee17449973edc9a991f517d90fcfd45d3e8
Bytes 244754233
```

Complete-stage seal SHA256:
`50fbac86d7a0078dd952148d80a25355458d78ea70526790d0561466e32ff3c7`.
Both are pinned in[outcome.json](outcome.json), with full source bindings.

## Stage2 admission — prepared only

**Admitted, not executed.** Observed worst-case training step andcomplete
validation-epoch timings, with unchanged1.5 factor+120s, project:

- Remaining4 epochs: **7693.057869s**, below10800s.
- Aggregate: **10390.490495s**, below18000s.
- Remaining aggregate allowance:15302.567374s.
- Projected RSS:4702602KiB (4.485GiB), below6144MiB; disk reserve passes.

[stage2_admission.json](stage2_admission.json) and[stage2_identity.json](stage2_identity.json)
are copies of the external admission files. The original files are at
`/home/user/epd-v2-artifacts/segmented-300-001/stage2-preflight`.
The later authorized run can continue with:

```bash
python3 tools/segmentation_v2/campaign.py train \
  --policy tools/segmentation_v2/campaign_segmented_300.json --stage 2 \
  --resume-stage /home/user/epd-v2-artifacts/segmented-300-001/stage1 \
  --resume-sha256 98bb7afde01d0c013df0c4e40ba8cee17449973edc9a991f517d90fcfd45d3e8 \
  --artifacts /home/user/epd-v2-artifacts/training-expansion-002/integration/dataset.json \
  --weights /home/user/epd-v2-artifacts/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth \
  --preflight /home/user/epd-v2-artifacts/segmented-300-001/stage2-preflight \
  --output /home/user/epd-v2-artifacts/segmented-300-001/stage2
```

This command was not executed. Keep source/dependency/dataset/recipe/policy hashes
unchanged. It starts epoch5/globalstep1201, restores SGD momentum andRNG, and
retains Stage1 metrics/checkpoints for selection across all8 epochs.

## Artifacts and scope

[artifact_locations.json](artifact_locations.json) pins all external artifacts:
992535702bytes (946.56MiB), including all4 checkpoints and48 raw predictions.
Local retained filesystem only; no remote durability claim. Raw scene images,
IDs andweights remain outsideGit. The original300-scene composite hash is
`04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11`.
Model initialization SHA256:
`bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0`.
All1200 loss records andCSV are retained inGit as metadata, with a derived
[Stage1 loss curve](loss_stage1.svg); original external curve is preserved unchanged
(its inherited caption says full8 epochs; actual data contains Stage1 only).

No held-out pixels, historical regression inference, ONNX export, production
adapter/association changes, EMD modifications, motion, merge ordeployment.
Renderer IDs were supervision/scoring only; inference received RGB exclusively.
Conservative geometry andthe0.1mm physical qualification requirement are unchanged.
Next milestone: a separately authorized run resumes epochs5–8 fromthe verified
checkpoint andapplies the same frozen gate after completing the experiment.
