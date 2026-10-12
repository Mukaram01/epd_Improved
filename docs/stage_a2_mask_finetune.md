# Original-model mask fine-tuning: validation improvement, Stage-A2 blocked

**BLOCKED_STRICT_ASSOCIATION.** One campaign completed; epoch2 passes frozen validation, but its two confident Stage-A2 instances remain rejected. No model promotion, ONNX export, C++ candidate parity, deployment or motion. This draft is stacked on #220; #219/#220 remain unchanged.

## One preregistered experiment

Start from original SHA256 `b2668971dd1230cdbc93bbc7d0182b46c91e7469717dc210052fa4eb20fa8bdb`. Strict-load all307 tensors; retain two-class heads, FrozenBN epsilon0, RGB256→INTER_LINEAR RGB512/255 and original internal800/1333 transform. Train the existing heads, frozen backbone, SGD lr0.0001/momentum0.9/weight-decay0.0001, no scheduler, mask loss×2 and other losses×1. Two epochs, batch1, FP32, two CPU threads, workers0, seed260110. The smallest scoped change emphasizes native supervised mask accuracy while preserving the recovered architecture and allowing detection heads to maintain/improve recall. No boundary correction, oracle crop, erosion or inference threshold change.

Reuse the verified300-scene dataset manifest SHA256 `04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11`:484 visible instances,50 negatives; same12 validation/12 sealed test memberships. Training verifies source artifact hashes through the existing loader. A Python audit hook rejects test-directory reads. Scene order is exactly the existing seeded epoch shuffle;600 contiguous updates and300 distinct scenes per epoch were verified from history. No Stage-A2 case enters training or model selection.

Admission reuses the retained original-model full-SGD benchmark from #220, rather than repeating it: same architecture, frozen backbone, batch/precision/threads and forward/backward structure; scalar loss weighting/LR differ explicitly. Maximum train4.888312534s/validation2.864535877s yields `(600*train+36*validation)*1.5+240 = 4794.166218s`. Projected RSS2.792GiB. Both pass unchanged10800s execution/18000s aggregate/6GiB RSS/5GiB reserve ceilings. Historical failed eight-epoch admission remains failed. No hyperparameter search or retry occurred.

## Actual campaign and frozen gates

Native exit0; full process2579.315612s (42.99min); training plus candidate validation2511.929033s, excluding initial baseline/startup. Peak RSS3,017,296KiB (2.878GiB), below6GiB; slightly above the2.792GiB projection, retained honestly. Exactly600 optimizer updates, two epochs. Complete losses, raw predictions and source snapshots remain external. Checkpoints atomically publish after validation and retain model/optimizer/RNG/epoch/step/config/source and history/validation hashes. No automatic continuation/retry.

| Model | Recall | Mean IoU incl misses | FP | Spill px | Cross-object px | Missing px | Mean boundary px | Outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Genuine original baseline |17/22|0.685890080|1|1915|914|8239|117.543601|Reference|
| Epoch1 |17/22|0.712738677|1|1184|683|6802|117.124329|Rejected: recall|
| Epoch2 |18/22|0.758658574|1|1185|654|5154|93.828403|Validation pass|

Cube precision/missed instances are baseline17/18 (94.44%)/5, epoch1 17/18 (94.44%)/5, epoch2 18/19 (94.74%)/4.

All three use confidence≥0.8 and mask probability strictly>0.5, identical RGB preprocessing and unchanged scorer. Frozen gates: recall≥18/22, IoU>0.685890, FP≤1, spill<1915, zero class/identity/output violations. All scoped output checks pass. Both negative scenes024/030 have zero confident detections for all three models. Boundary means include unchanged512px miss penalty. Per-instance metrics, every detection and genuine raw predictions are retained; no held-out test evaluation. Epoch2 is the only eligible checkpoint, selected by the unchanged validation ranking.

Checkpoint references (outside Git):

- Epoch1: `/home/user/epd-v2-artifacts/mask-finetune-001/epoch_01.pt`,244737507 bytes, SHA256 `9a2a20377b6b9938da5b00840d4681ee0cf1b159879f80ff34340c340460f848`.
- Epoch2: `/home/user/epd-v2-artifacts/mask-finetune-001/epoch_02.pt`,244737507 bytes, SHA256 `f3e7531c25cb20920c04283d205d7241028d7321b1ea34e2f1848415744a3b56`. **Validation-qualified only; Stage-A2 rejected.**

## Genuine Stage-A2 check: decisive blocker

Only after validation selection, run exactly one bounded CPU PyTorch inference on the retained paired-native two-cube case. RGB SHA256 `92f1ba8ad58b2437fa5ccf55e73d53cc9dcba469b04efefbe624ce1505585793`. Losslessly extract retained gzip artifacts and verify their original manifest hashes. Reuse the existing read-only `load_capture`, `map_mask` and `associate` functions without modifying EMD. The complete capture/ownership provenance passes. IDs are evaluation references only; the model receives RGB alone.

Use integer-clipped boxes and exact full-image ROI slices, matching the existing EPD adapter, then the unchanged conservative union2×2 map to256. No second mask resize, fitted offsets, trimming or threshold modification. This is actual unchanged strict association against genuine PyTorch predictions; **not a claim of candidate ONNX/EPD C++ consumption**.

| Metric (two confident cubes) | Retained original | Epoch2 candidate |
|---|---:|---:|
| Native512 raw background pixels |456 /635|310 /572|
| Native512 ROI background pixels |303 /564|310 /526|
| Conservative256 mapped background pixels |123 /160|**154 /160**|
| Mapped cross-object pixels |0 /0|0 /0|
| Strict accepted instances |0|**0**|

Candidate scores0.987813/0.968878; two additional detections0.137577/0.094547 remain below0.8 and are rejected. All four associations reject; neither confident mask is pure. Native exit2 intentionally records `BLOCKED_STRICT_ASSOCIATION`, wall8.780097s, no resource termination. Validation improvements do not solve the proven Stage-A2 grasp-pipeline blocker. Original comparisons are retained independent baseline evidence; original inference was not repeated. No further experiment, export or candidate C++ check follows this failure.

## Reproduce and retained evidence

```bash
python3 tools/segmentation_v2/mask_finetune.py run \
  --checkpoint /home/user/epd-v2-artifacts/original-cube-checkpoint/cube_maskrcnn_checkpoint_thresh_08.pth \
  --artifacts /home/user/epd-v2-artifacts/training-expansion-002/integration/dataset.json \
  --output /home/user/epd-v2-artifacts/<new-exclusive-campaign-directory>
python3 tools/segmentation_v2/stage_a2_mask_check.py run \
  --campaign /home/user/epd-v2-artifacts/mask-finetune-001 \
  --retained /home/user/workcell_ws_stage_a2/evidence/stage_a2_geometry/physics_owner/same_fragment/paired_native_reference \
  --emd-scripts /home/user/workcell_ws_stage_a2/scripts \
  --output /home/user/epd-v2-artifacts/<new-exclusive-association-directory>
```

Commands document reproduction; no second run is authorized/executed here. The Stage-A2 checker refuses a checkpoint that did not pass validation or whose SHA changed. Original model, data and prior evidence remain protected.

Two targeted tests pass (mask-only loss weighting, frozen gates and conservative admission); red/green evidence is retained. The genuine association check exercises the unchanged consumer and records its failure. Focused read-only review found no demonstrated issues in either runner or post-selection check. Existing ROS/GPU collection blockers were not rerun or changed. Artifact locations/hashes, executed sources, losses, per-instance reports and raw predictions are in `evidence/segmentation_v2/mask_finetune_001/artifact_locations.json`; large files stay outside Git.

**Single next action:** authorize one boundary-weighted mask-loss experiment targeting residual native boundary spill, while preserving the improved validation recall gate and requiring genuine Stage-A2 strict acceptance. No further training is performed in this task. Conservative geometry, fail-closed association and0.1mm physical-contact requirements remain unchanged; no physical/grasp qualification is claimed.
