# Original cube-trained Mask R-CNN recovered

**ORIGINAL_MODEL_RECOVERED**. This recovers the original checkpoint and demonstrates one-input numerical compatibility with the unchanged original ONNX. It is not a new quality, held-out, C++ or production qualification.

## Checkpoint and provenance

`/home/user/epd-v2-artifacts/original-cube-checkpoint/cube_maskrcnn_checkpoint_thresh_08.pth`

- 176226850 bytes; SHA256 `b2668971dd1230cdbc93bbc7d0182b46c91e7469717dc210052fa4eb20fa8bdb`.
- Loaded only with CPU `torch.load(weights_only=True)`, no allowlist or unsafe fallback. Original file and ONNX hashes are verified unchanged.
- Self-reported metadata: architecture `maskrcnn_resnet50_fpn`,2 classes (`background`, `cube`), torchvision `0.26.0+cu130`,3 completed epochs,114 training images,11 validation images. These are declarations, not authenticated dataset/source identities.
- No optimizer, scheduler, RNG, seed, learning rate, freeze policy, original data hashes or training/export source exists in the supplied checkpoint/directory. Exact historical training cannot be resumed or reproduced from it alone.

## Architecture and original ONNX identity

ResNet50 (3/4/6/3 bottleneck blocks),256-channel FPN,53 FrozenBatchNorm modules with **epsilon0**. Four256-channel mask convolutions,256-channel2x2 transposed convolution and2-channel1x1 mask logits produce the instance mask head. Classifier weight `[2,1024]`, box regressor `[8,1024]`, mask logits `[2,256,1,1]`; class0 background,class1 cube. Existing label-file `__background__` is the background alias.

All307 state tensors load strictly without missing/unexpected keys or head replacement. All201 ONNX initializers are accounted for:95 named weights bitwise equal,106 folded FrozenBN scale/bias constants numerically matching (rtol/atol1e-6; maximum absolute difference3.814697265625e-6). The ONNX producer is PyTorch2.11.0, opset17; its SHA is `21362d62d816bd684f2b5c7769d0bcd1cd568af86419788fca9d095c46f2bad2`.

The bundled EPD GUI training/export scripts use `maskrcnn_benchmark`, legacy model paths/config and opset10. They are not the source of this torchvision checkpoint/opset17 graph. Their hashes are retained in `legacy_source_identity.json`; no unsafe legacy loader/exporter was executed. No actual cube-training/export script was found in the supplied directory or current repository/workspace Python sources searched. Initialization ancestry is not fully established; epsilon0 behavior agrees with torchvision COCO_V1 construction, but the original initialization file/hash is absent.

External input is genuine RGB256 -> OpenCV INTER_LINEAR RGB512 -> NCHW float32/255. The model internally normalizes mean[.485,.456,.406],std[.229,.224,.225] and resizes with min800/max1333. Outputs are boxes float32[N,4],labels int64[N],scores float32[N],full-image probabilities float32[N,1,512,512]. Torchvision's internal score prefilter is.05 and detection cap100; the checkpoint filename does not imply a baked.8 filter. EPD confidence remains>=.8 and mask foreground strictly>.5. No extra mask resize, trimming, erosion or box corrections are used.

## One actual bounded CPU parity test

Retained frozen validation scene025 uses hash-pinned `adapter-historical-001/historical_bgr512.png`, exactly the earlier RGB512 input, with its retained original ONNX outputs. No ID raster, held-out pixels or historical two-cube regression input was opened. One PyTorch inference and one ONNX inference ran; neither was repeated. Installed Torch2.7.0/torchvision0.22.0 were used without dependency changes.

| Comparison | Boxes max abs | Scores max abs | Masks max abs | Labels/count | >.5 decision changes | >=.8 decision changes |
|---|---:|---:|---:|---|---:|---:|
| PyTorch vs original ONNX |0|0|0.000006407499313354492|Exact,1 instance|0|0|
| New ONNX vs retained confident reference |0|0|0|Exact,1 instance|0|0|

Predeclared absolute tolerances: boxes1e-3,scores1e-5,probabilities1e-4; labels/count and foreground/confidence decisions exact. Same descending-score order corresponds on this input. General correspondence across multiple inputs and production qualification are not claimed.

### All failures retained

1. Attempt001 stopped at static identity before any model inference: the initial reconstruction used FrozenBN default epsilon1e-5. The graph disagreed by0.00658183 in its first scale. Inspection of graph constants and installed COCO_V1 construction established epsilon0. A regression test was watched fail then pass. No original weight/threshold was changed.
2. Attempt002 completed both genuine inferences and saved raw predictions, then exited1 in reporting because the historical NPZ stores `masks_confident`, not raw `masks`. This was a report-schema bug, not a numerical mismatch. The code now explicitly filters original boxes/labels/scores at>=.8 and uses the already-confident masks without altering probabilities. A regression test was watched fail then pass.
3. An offline-only audit hash-verifies both saved predictions, original inspection/process and exact executed-source snapshot, then produces `original-cube-recovery-004/result.json`. No inference was rerun to obtain a clean process exit. The original native exit1 remains recorded. Offline audit exited0. The total inference-worker process took11.983485869s, peak2063604KiB, within180s/6GiB. Individual model durations were lost with the report failure and are not fabricated. The earlier static check took5.426005693s with no inference.

The independent review reproduced the saved comparisons and found that failed parity could exit successfully. Worker and offline-audit paths now return nonzero for blocked verification; the regression test failed before the fix and passes after it. All44 focused tests pass. Repository pytest stops during collection at the inherited missing `launch.actions` and GPU-only GUI errors; its two configuration side effects were restored. No test bodies ran in that repository-wide check. No environment-wide upgrades, new model downloads, training, checkpoint rewriting or model export occurred. Large checkpoint/ONNX/raw predictions remain outside Git with paths/hashes in `artifact_locations.json`.

## Exact next training milestone

Create and resource-admit **one versioned fine-tuning campaign from this original hash-pinned checkpoint**, using the existing composite300-scene training loader and original12-scene validation gate. This is a new fine-tuning experiment with fresh optimizer/RNG, not resumption of the unknown original optimizer. Do not initialize from COCO or rejected v2 weights, and do not replace either original two-class head. Explicitly preregister optimizer/scheduler and freeze policy because the original ones are unknown.

Use existing `train.tensors` to supply RGB512/255 and genuine native512 training targets, then let the reconstructed original model apply its established normalization and800/1333 transform to images/targets. The loader's class1=cube mapping already agrees. Preserve the300/12/12 memberships, >=.8 confidence, >.5 mask threshold and frozen recall/IoU/FP/spill gate. Measure CPU throughput/memory for the800-pixel internal transform before admitting the future bounded campaign; the rejected replacement used a different512 internal transform, so its resource estimate cannot be reused blindly. No held-out/historical tuning.

This task stops before that campaign. PR219 remains draft. EMD PR3176, production adapters/association, conservative geometry, collision/contact0.1mm requirements, controllers, motion and deployment remain unchanged.
