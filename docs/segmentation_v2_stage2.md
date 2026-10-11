# Stage 2 and final model selection

**V2_300_EIGHT_EPOCHS_COMPLETED_REJECTED**. No checkpoint qualifies; no model is promoted.
The strongest diagnostic is epoch 8 using the unchanged ranking, and remains rejected.

| Epoch | Mean loss | Recall | Mean IoU | FP | Spill px | Cross px | Missing px | Boundary px | Failed gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Historical baseline | — | 17/22 | 0.685890 | 1 | 1915 | 914 | 8239 | 117.543601 | Reference |
| 1 | 0.624834 | 9/22 | 0.376874 | 8 | 51125 | 1372 | 24806 | 258.021066 | recall, iou, false_positives, background_spill |
| 2 | 0.437894 | 9/22 | 0.363739 | 4 | 607 | 1186 | 26716 | 257.796601 | recall, iou, false_positives |
| 3 | 0.372394 | 10/22 | 0.434413 | 1 | 634 | 226 | 23465 | 256.919430 | recall, iou |
| 4 | 0.343069 | 11/22 | 0.435827 | 1 | 574 | 2687 | 27909 | 256.955429 | recall, iou |
| 5 | 0.309075 | 12/22 | 0.472580 | 1 | 1097 | 1354 | 19063 | 233.756070 | recall, iou |
| 6 | 0.280181 | 9/22 | 0.398234 | 2 | 935 | 425 | 25797 | 279.997620 | recall, iou, false_positives |
| 7 | 0.273532 | 12/22 | 0.504904 | 5 | 823 | 1543 | 19590 | 210.724461 | recall, iou, false_positives |
| 8 | 0.259110 | 15/22 | 0.561233 | 3 | 1081 | 2692 | 14825 | 164.436940 | recall, iou, false_positives |


Confidence >=0.8; mask probabilities strictly >0.5. Each epoch fails recall and IoU.
Epochs 1, 2, 6, 7 and 8 additionally fail FP; epoch 1 additionally fails spill.
All epochs have zero scoped class/output contract violations. This is not EMD association or physical qualification.
Both negative validation scenes (024 and 030) have zero confident detections in every epoch (16/16 negative evaluations).
Boundary means preserve the scorer's 512px missing-boundary penalty; per-instance values, detections and every failure are in the epoch summaries.

Stage 2 resumed the pinned epoch-4 state at step1201, executed exactly1200 updates,
and ended at epoch8/step2400. The independent auditor verifies exact seeded order,
300 unique samples per epoch, all eight checkpoint model/optimizer/RNG/config identities,
96 raw prediction hashes and exact per-instance/detection/negative metric replay.
The 300/12/12 frozen dataset remains unchanged: train484 visible instances/50 negatives.
No held-out pixels or historical regression predictions were accessed.

Stage1 worker wall: 2697.432626146s; Stage2: 2796.566909145s.
Aggregate: 5493.999535291s; peak RSS: 3531208KiB (3.367622GiB).
Both native exits are0, no resource termination. Bounds remain10800s/stage,
18000s aggregate,6GiB RSS,5GiB disk reserve. Preflight/audit/test overhead is separate
from worker accounting. The historical failed full-run feasibility gate is unchanged.
The post-Stage2 `stage2_admission` auxiliary field is not authorization for a third stage.

Before launch the clean requested baseline e5a8ca6a9005f942cfa1ab67cf3565b7064c3497,
all94 retained artifact hashes, Stage1's81 sealed files, epoch4 SHA, dataset,
COCO initialization, source/dependency binding and existing feasible=true admission
were verified. No source files changed before the exact authorized command launched.
The ten execution sources, recipe and policy remain unchanged after execution.
Stage2's executed source snapshots and process command are sealed outside Git.
Audit-only code/tests were added after launch.

`checkpoints.sha256` lists all eight full digests. `artifact_locations.json` lists
184 immutable external files (1977486942 bytes), including all checkpoints,
raw predictions, source snapshots, stdout/stderr and stage/admission evidence.
Storage root: `/home/user/epd-v2-artifacts/segmented-300-001`. Files are local external artifacts, not remotely backed up by this PR.
Large weights, images, ID rasters and predictions remain outside Git.
Stage1/pilot/rejected prior campaigns and hashes remain unchanged.

Verification:37 focused deterministic tests passed, including actual-Torch continuous8
versus4+4 state equivalence, corrupt/incompatible checkpoints, repeated/missing steps,
seal mutations and rejection-only selection. This does not claim a second uninterrupted
Mask R-CNN campaign was executed. `audit.json` contains the genuine campaign replay.

Reproduce the offline audit (new exclusive output path required):

```bash
python3 tools/segmentation_v2/audit_segmented.py \
  --artifacts /home/user/epd-v2-artifacts/training-expansion-002/integration/dataset.json \
  --stage1 /home/user/epd-v2-artifacts/segmented-300-001/stage1 --stage2 /home/user/epd-v2-artifacts/segmented-300-001/stage2 \
  --resume-sha256 98bb7afde01d0c013df0c4e40ba8cee17449973edc9a991f517d90fcfd45d3e8 \
  --output /external/unused-audit.json
```

Next milestone: a separately authorized, preregistered fine-tuning revision targeting
missed objects and false positives, retaining the frozen gate and held-out isolation.
No further training, export, C++ candidate parity, deployment, motion, EMD changes
or merge was performed. PR219 stays draft.

Repository-wide pytest exits2 during collection: `test_epd_emd_pipeline_launch.py` lacks `launch.actions`; `gui/test_gui_gpu_local_only.py` exits1 because no NVIDIA device exists. These unchanged baseline files are not Stage2 regressions; no dependency or hardware workaround was attempted. All six CI runs for the exact starting baseline e5a8ca6 already failed; `base_ci.json` retains their identities and links. New-head CI status is reported separately after push.

Repository pytest collection changed two deployment config files as a test import side effect. Both were restored exactly to the clean baseline; the committed change contains no deployment config edits. Fresh read-only review found no issues.
