# One 300-update CPU experiment: learned 56×56 mask logits

**Status: runner implemented only; no campaign executed or candidate qualified.**

This branch is stacked on PR #223 (four disposable training updates, 56-grid
model/EPD API feasibility). It is a strictly bounded next experiment. It is
not the historical `mask_resolution.py` probe and must not be substituted
for that code.

## Prerequisites and immutability

Working branch `codex/epd-56-grid-one-epoch-campaign` under a clean worktree.
The original `~/epd_ros2_ws/src/easy_perception_deployment` checkout can
contain local C++ modifications: do not reset, stash or switch it.

Pinned inputs, externally stored in the existing Ubuntu environment:

- #221 epoch2 checkpoint
  `/home/user/epd-v2-artifacts/mask-finetune-001/epoch_02.pt`,
  SHA-256 `f3e7531c25cb20920c04283d205d7241028d7321b1ea34e2f1848415744a3b56`.
- Frozen dataset
  `/home/user/epd-v2-artifacts/training-expansion-002/integration/dataset.json`,
  SHA-256 `04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11`.
- The original failed and resumed disposable feasibility process/result JSON
  files in `evidence/segmentation_v2/mask_resolution_001`, whose report
  must verify 4 disposable optimizer steps, 0 full campaign steps, original
  mask-resolution source hash, `17.902557s` worst-case bound and `2.951GiB`
  projected RSS. The failed process must remain visible: it is never
  rewritten into a successful or timed sample.

Admission is resource-bounded to exactly one epoch and 300 distinct
train-scene updates, 12 frozen validation scenes, two CPU threads, no CUDA,
6GiB RSS process cap, 10,800s wall cap, at least 5GiB remaining disk plus
1GiB output allowance. The original projection is **8,940.64s**; actual
training speed and whole-process wall time are still unknown. A focused
local model shape test took 247.827s, so account for possible cold-start
overhead. The resource projection is evidence, not a performance guarantee.

## First: offline tests and no-write preflight

```bash
cd /home/user/epd-v2-supervised-pilot
git status --short --branch
python3 -m unittest discover -s tools/segmentation_v2 \
  -p 'test_mask_resolution_campaign.py' -v

python3 tools/segmentation_v2/mask_resolution_campaign.py check \
  --output /home/user/epd-v2-artifacts/mask-resolution-campaign-001
```

The proposed **output directory must not already exist**. `check`
verifies real current checkpoint/dataset hashes, frozen code/evidence, original
one-epoch time/memory projection and current available disk/CPU. It prints
`GO_ONE_EPOCH_ONLY` or `BLOCKED` as JSON; does not train, write a
checkpoint or evaluate held-out/Stage-A2 pixels.

**Do not run the costly campaign** if these tests or preflight fail.

## Next: one campaign, only after reviewing preflight

```bash
cd /home/user/epd-v2-supervised-pilot
python3 tools/segmentation_v2/mask_resolution_campaign.py run \
  --output /home/user/epd-v2-artifacts/mask-resolution-campaign-001
```

This creates the output directory **exclusively** and writes admission,
policy/source/dependency bindings and exact executed-source copies. A bounded
owned subprocess trains one epoch of 300 unique verified training scenes.
Its only architecture change is ROIAlign 14×14→28×28, producing 56×56
learned mask logits with all original 307 tensors loadable. Freeze all but
the 12 mask-head/predictor parameters; ordinary mask BCE×2 and original
optimizer (SGD LR 0.0001, momentum 0.9, weight decay 0.0001), same seed
260110, class map, FrozenBN, FP32 and preprocessing. No boundary-weighted
loss, threshold adjustments or retraining loops.

Each actual update is recorded in `history.jsonl`, fsynced before the next
step; the completed 300-scene order must match the frozen seed and contain no
repeated/missing scenes. Verification rejects changes to frozen non-mask
tensors/buffers before validation. The sole completed checkpoint
`epoch_01.pt` is published atomically with model, optimizer, RNG, 56-grid
architecture, experiment config, immutable source and validation hashes.

A killed, interrupted, corrupt or over-budget run stays failed with preserved
logs/process/partial history; it is **never silently resumed or reattempted**.

## Validation gate and decisive next action

The existing `campaign.validate_epoch` checks all twelve frozen validation
scenes and both negatives, with unchanged thresholds:

- Recall ≥18/22
- Mean IoU >0.685890
- False positives ≤1
- Background spill <1915
- Contract violations =0
- Detection confidence ≥0.8; mask probability strictly >0.5

Output `result.json` is `VALIDATION_PASS` with a selected SHA-pinned
checkpoint, or `TRAINED_REJECTED` with a completed-but-ineligible checkpoint.
Both mean **no Stage-A2 test yet**. Rejection is a correct completed
experiment, never a reason to automatically retry. No sealed held-out pixels
are accessed.

Only after `VALIDATION_PASS`, separately run the existing (adapted)
`stage_a2_mask_check.py` against the retained EMD two-cube capture and frozen
unchanged association contract. The checker explicitly reconstructs the
56-grid architecture when and only when the checkpoint records its exact
architecture/policy/completion identity; older 28-grid checkpoints remain
supported. Require **2/2 strict associations**. Never allow renderer IDs into
model inference, artificial mask editing/erosion, adjusted boxes or altered
confidence/association thresholds.

**If Stage-A2 fails:** preserve evidence, stop, and do not export/promote.
If Stage-A2 passes: separately qualify ONNX export, ORT numerical/foreground
parity, zero-detection runtime and actual EPD C++ CPU ingestion. Do not access
held-out pixels or claim motion/physical grasp qualification from this run.

## Boundaries

No EMD source modifications, EPD production model replacement, camera
launch, Gazebo capture, robot motion, ROS runtime, deployment or CI-based
qualification occur in this PR. The code was authored through GitHub rather
than executed on the user's Ubuntu system. Tests and real training are
**pending independent workstation execution**.
