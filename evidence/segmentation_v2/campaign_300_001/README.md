# 300-scene campaign preflight: resource stop

**BLOCKED_TRAINING_RESOURCES**. Starting commit
`e3306a64bf0cabf25b982572ff517f9f6245b409`; existing draft PR219.
No training campaign launched: **0 optimizer steps,0 epochs,0 new checkpoints**.
No epoch validation/model selection, ONNX export or C++ candidate parity executed.
No held-out pixels or historical regression predictions accessed.

The real composite loader verified300 training scenes,484 visible instances and
50 negative scenes, plus12 frozen validation scenes. All3120 train/validation
artifact hashes passed; native ownership audit passed for312 scenes. All324
split memberships were checked from metadata only. Original campaign and smoke
checkpoint hashes were reverified without loading them. No dependency changes.

## Measured feasibility

CPU only:8 logical CPUs, PyTorch2.7.0 /torchvision0.22.0,2 training threads;
CUDA build12.6 but0 available CUDA devices. Available RAM11.46GiB;
free disk84.81GiB before measurement. Pinned fresh COCO_V1 initialization;
seed260110, unchanged heads/SGD/frozen-backbone recipe.

| Forward/backward scene | Objects | Seconds |
|---|---:|---:|
| scene_000 (negative) |0|2.285832|
| expansion_122 |3|2.348129|
| expansion_093 |3|2.044599|
| expansion_123 (smallest visible foreground) |1|4.033226|

These are real forward/backward loss computations **without optimizer updates**,
not a smoke checkpoint or training campaign. Loss components are retained in
[feasibility.json](feasibility.json). Validation timing-only forwards onscene_024
andscene_026 took5.557282s and2.041656s; they were not scored or used for selection.

Unchanged conservative projection:
`(2400 * max_train_seconds + 96 * max_validation_seconds) * 1.5 + 120`
= **15439.860741s (4h17m20s)**, exceeding10800s by4639.860741s.
Measured peak RSS1746548KiB (1.666GiB), projected2619822KiB (2.499GiB),
below6144MiB. Disk check reserved5GiB plus8*400MiB checkpoint allowance.
The wall gate failed; no retry, altered factor, shortened run or budget override.
Actual future training duration remains unmeasured, not asserted from this estimate.

There are no epoch losses/validation metrics, best checkpoint/hash or ONNX/hash
for this attempt. Existing rejected checkpoints are preserved, not reused.

## Implementation and verification

- `campaign.py --policy campaign_300.json` accepts the hash-verified composite
  manifest, enforces300/12 counts and unique/disjoint split IDs; original directory
  mode/default24 policy remains supported. Train recipe and original campaign
  configuration are unchanged. Every epoch uses all300 samples with seeded order.
  The full runner ignores the legacy smoke-only cpu_max_steps field; its full
  step count comes from the explicit policy and verified loader length.
- Every epoch saves model, optimizer and Python/NumPy/Torch/CUDA RNG states,
  append-only step losses and a mean-loss/validation summary. Existing finalCSV/SVG
  generation remains available after full completion. These paths are prepared,
  **not claimed executed** on300 scenes.
- `audit_campaign.py` supports24 or300 sample counts and replays hash-pinned
  validation outputs/selection without model inference.
-29 focused deterministic tests pass; compileall andgit diff --check pass.
  Starting-head GitHub CI has six failed jobs (two humble_ci and four
  python_ci); exact URLs/statuses are retained in[ci_starting_head.log](ci_starting_head.log).
  No green CI claim is made. The broader ROS/C++ build was not rerun: this change only affects offlinePython
  tools, and no model qualified for C++ candidate verification.

## Reproduce after provisioning resources

Run from the repository with existing pinned dependencies. Use a new exclusive
external output directory, never overwrite this failed preflight. Wrap measurement
with existing `pilot.bounded` (180s,6144MiB) when orchestrating it:

```bash
python3 tools/segmentation_v2/measure_300.py \
  --artifacts /home/user/epd-v2-artifacts/training-expansion-002/integration/dataset.json \
  --weights /home/user/epd-v2-artifacts/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth \
  --output /external/new-resource-preflight
# Only after feasible=true and current source hashes are pinned:
python3 tools/segmentation_v2/campaign.py train \
  --policy tools/segmentation_v2/campaign_300.json \
  --artifacts /home/user/epd-v2-artifacts/training-expansion-002/integration/dataset.json \
  --weights /home/user/epd-v2-artifacts/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth \
  --preflight /external/new-resource-preflight --output /external/new-campaign
```

The rejected preflight cannot authorize current code: executed sources are
hash-pinned inidentity.json and archived externally; `executed_measure_300.py`
is the exact measured script. The final reusable script adds portable CLI inputs
and CUDA timing support; final runner adds explicit split checks and requires matching-device
feasibility for both CPU and CUDA after the resource
stop. Neither was remeasured or launched. A future environment must measure current
sources before training; this prevents reusing stale feasibility approval.

Dataset SHA256: `04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11`.
Initialization SHA256: `bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0`.
External evidence: `/home/user/epd-v2-artifacts/campaign-300-001-preflight`;
[artifact_locations.json](artifact_locations.json) pins every retained file.
Local retention only; no remote artifact durability claim.

**Next milestone:** provision CUDA-capable hardware and pass a fresh measured
feasibility check, then run the single300-scene campaign under the same gate.
Current CPU needs at least1.43x measured throughput for the same conservative
10800s projection. No model qualification, EMD modification, deployment or motion.
