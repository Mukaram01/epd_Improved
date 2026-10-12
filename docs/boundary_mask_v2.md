# Boundary-weighted mask experiment: rejected by Stage-A2

**BLOCKED_STRICT_ASSOCIATION.** One admitted experiment completed; frozen validation passes, but Stage-A2 accepts **0/2**. No promotion, export, candidate C++ check, second experiment, deployment or motion. Stack this draft on #221 (`f04de31`); preserve #219–#221 and EMD #3176.

## Diagnosis reused without repeating inference

Retained Stage-A2 `onnx_head_observation` and `onnx_mask_static` evidence shows overlapping causes, not exclusive learned-head attribution. Actual foreground over background has interior sigmoid support; 130/62 background pixels have every positive-weight interior contributor above0.5, and 55/29 high head sample centers map to nearest native background. Fixed-box bilinear versus nearest changes6/166 background decisions. Zero padding suppresses foreground rather than creating it; placement/padding causes75 missing pixels,72 bounded below0.5 even with all interior values1. These raster/counterfactual observations do not prove continuous geometry or that nearest sampling is correct.

Observed ONNX outputs match retained original outputs; #221 PyTorch masks also spill. An ONNX-only failure is ruled out. EPD's full-image-to-ROI slicing retains exact values without another resize and can only remove pixels. A learnable boundary contribution is supported; box placement and resolution remain possible limitations. No inference correction was attempted.

## Preregistered experiment and admission

Start from #221 epoch2 SHA256 `f3e7531c25cb20920c04283d205d7241028d7321b1ea34e2f1848415744a3b56`, derived from the protected original `b2668971dd1230cdbc93bbc7d0182b46c91e7469717dc210052fa4eb20fa8bdb`. Fresh optimizer/RNG; exactly one epoch/300 updates. Freeze everything except existing mask head/predictor. Keep architecture,307 strict-load tensors,two classes,FrozenBN epsilon0,RGB256→linear512→CHW/255 and original internal800/1333 transform.

At the existing28×28 training target, weight BCE4× where a3×3 max/min neighborhood varies, including zero-padded outside support; normalize by total weights. Targets remain unchanged, including soft projected targets. Existing overall mask-loss multiplier2 remains. This is training-only weighting, never runtime erosion, oracle cropping or probability correction. SGD lr0.0001,momentum0.9,weight-decay0.0001,no scheduler,batch1,FP32,two CPU threads,workers0,seed260110; see `tools/segmentation_v2/boundary_mask.json`.

Use the same300 training scenes/484 instances/50 negatives and12 validation scenes. Manifest SHA256 `04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11`. Held-out12 pixels untouched; Stage-A2 excluded from training/selection. Reuse hash-pinned baseline predictions instead of repeating inference. The loader checks original per-file bindings.

Admission reuses the retained full-SGD benchmark: same forward, smaller mask-only backward subset;28-grid weighting cost covered by unchanged1.5 safety factor. Conservative projection2542.864s including24 validation forwards and240s overhead,2.792GiB RSS,5GiB disk reserve. Existing10800s execution/18000s aggregate/6GiB limits unchanged; historical failed eight-epoch admission remains failed. Actual process exit0,1024.300s total; training plus validation1014.322s; peak2,713,712KiB (2.588GiB). All300 distinct scenes in exact seeded order, steps1–300; no retry.

## Identical validation and genuine Stage-A2 results

| Model | Recall | IoU incl misses | FP | Spill | Missing | Cross-object | Mean boundary px |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original |17/22|0.685890080|1|1915|8239|914|117.543601|
| #221 epoch2 |18/22|0.758658574|1|1185|5154|654|93.828403|
| Boundary candidate |18/22|0.759171776|1|1330|4896|750|93.856551|

Candidate passes unchanged recall≥18/22,IoU>0.685890,FP≤1,spill<1915,violations0. Confidence≥0.8 and mask strictly>0.5. Precision18/19; four missed instances; both negative scenes024/030 have zero confident detections. Boundary includes existing512px miss penalty. Per-instance/detection results and all raw predictions are retained. Validation spill, cross-object contamination and boundary mean worsen relative to #221 despite slightly better IoU/missing coverage; no claim of general improvement.

| Stage-A2, two confident objects | Original retained | #221 retained | Boundary |
|---|---:|---:|---:|
| Native512 raw spill |456/635|310/572|270/568|
| Native512 ROI spill |303/564|310/526|270/502|
| Conservative256 mapped spill |123/160|154/160|144/157|
| Strict accepted |0/2|0/2|**0/2**|

Candidate mapped missing0/0,native512 raw missing5/0,cross-object0/0. Scores0.987813/0.968878; other two detections remain below0.8. The unchanged authoritative association rejects both confident masks as mixed/missing/ambiguous IDs and both low-confidence detections. Native association exit2 is a genuine qualification failure, not a resource failure (7.676s,peak1,349,008KiB). IDs are offline references only; model input is genuine RGB. Integer-clipped ROI extraction and unchanged conservative union2×2 mapping apply without mask modification. No historical/sealed held-out inference.

## Evidence, reproduction and stopping point

Candidate external checkpoint `/home/user/epd-v2-artifacts/boundary-mask-001/epoch_01.pt`:186734847 bytes,SHA256 `22bf99d95e00e590aebb469bf77d20b7e1cbb3d8065a5a0e59a86c8f5291f305`, **rejected**. Contains optimizer/RNG/config/source/history/validation bindings, atomically published. Original and #221 checkpoints unchanged. Audit proves only12 mask tensors changed; boxes,labels,scores exactly match #221 on all12 validation inputs and Stage-A2. Execution sources match retained snapshots.

`evidence/segmentation_v2/boundary_mask_001/` retains results,per-instance validation,admission,process exits,audit and focused tests. `artifact_locations.json` lists105 external files with hashes/size/path (194483484 bytes), including weights,loss history,source snapshots and raw predictions. Four focused tests pass (2.544s); retained red test shows missing implementation before addition. Narrow read-only review found no demonstrated issues. No full repository/ROS/GPU collection rerun.

Commands: use existing `mask_finetune.py run` with `--policy tools/segmentation_v2/boundary_mask.json`, original checkpoint, composite dataset and exclusive output; then existing `stage_a2_mask_check.py run` on that campaign and retained paired-native capture. Exact executed arguments are retained in process JSON. Do not rerun this rejected experiment automatically.

**Single next milestone:** separately authorize a native-resolution mask-head experiment targeting residual Stage-A2 spill under the same gates. Current bounded task ends here. Fail-closed association,conservative collision geometry and0.1mm physical-contact requirements are unchanged; no physical qualification is claimed.
