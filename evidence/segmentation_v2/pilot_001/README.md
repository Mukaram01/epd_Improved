# Genuine pilot 001 — acquired, CPU smoke trained, candidate rejected

EPD base: `432a15317ff899fa8636533bf24dc44369b360b0`.
Read-only EMD baseline: `e25d03831292098543c2ed3467d50c65ce000762`, PR #3176.
Execution: 2026-10-11. No production or physical-contact qualification.

## What actually ran

- Pinned native infrastructure configured and compiled successfully, then the
  EPD-owned native capture configured/compiled in **14.47 s**, peak child RSS
  **894,608 KiB**. No dependency installation or broad environment upgrade.
- **48/48 genuine scenes** acquired once each, **24 train / 12 validation /
  12 frozen test**, with **4 / 2 / 2 negatives** and **43 / 22 / 21 visible cube
  instances**. Total acquisition **137.74 s**; scene times **2.62–5.43 s**;
  peak child RSS **629,152 KiB**. Native exits were all zero. No capture retries.
- Exactly **one** CPU training attempt, **two optimizer steps**. Training loop
  **15.85 s**; supervised process **22.60 s**; peak RSS **1,233,892 KiB**.
  Step 1 was a genuine negative image (mask loss 0). Step 2 was positive:
  mask loss **1.70160997**, total loss **2.84557295**. Checkpoint, optimizer state,
  exact executed source/config, RNG seed, history and curves are retained.
- Baseline ORT and smoke PyTorch predictions compared on **12 validation scenes
  only**. Initial diagnostic failed because the closure-captured model remained
  in training mode; all logs and baseline outputs are retained. One corrected
  bounded diagnostic completed in **68.62 s**, peak RSS **2,699,324 KiB**.
  No training rerun, selection, held-out prediction or historical v2 regression.

## Baseline versus v2 smoke — genuine validation predictions

Both receive the same RGB256 -> INTER_LINEAR512 -> float32 RGB CHW /255 input.
Both use confidence **>=0.8** and unchanged mask **>0.5**. Recall requires a
one-to-one match at IoU>=0.5. Misses receive IoU=0/boundary=512 pixels.

| Validation metric | Baseline ONNX | V2 CPU smoke |
|---|---:|---:|
| Recovered visible cubes / 22 | 17 / 22 | 0 / 22 |
| Detection recall | 77.27% | 0% |
| Mean instance IoU, including misses | 0.685890 | 0.084973 |
| Mean symmetric boundary error, including miss penalty (px) | 117.543601 | 309.015999 |
| False positives at gate | 1 | 40 |
| All confident detections | 18 | 40 |
| Background spill, sum over all confident masks (px) | 1,915 | 233,007 |
| Cross-object contamination, sum over all confident masks (px) | 914 | 1,679 |
| Missing visible pixels, sum over one-to-one object matches/misses | 8,239 | 51,850 |
| Native512 pixel purity accepted / rejected | 0 / 18 | 0 / 40 |
| **Authoritative EMD strict association accepted/rejected** | **NOT EXECUTED** | **NOT EXECUTED** |
| Prediction + retention time for 12 images (s) | 24.05 | 36.97 |

Per-detection totals include false positives and negative images; they may count
the same target more than once. Per-object misses above are separate. All
per-instance boundary/IoU values and all confident detection measurements,
including failures, are in [validation_report.json](validation_report.json).
Raw boxes, labels, scores and every mask used for measurement are retained in
hashed external NPZ files. Offline reanalysis reproduced the retained metrics
exactly; it never changes predictions or runs new inference.

**The smoke checkpoint is rejected as an improved model.** This is evidence that
the supervised path runs, not that two CPU steps improve segmentation. It was
never eligible for checkpoint selection, export or held-out evaluation.

## Artifact identity and failure evidence

Large raw datasets, binaries and weights stay outside Git. The exact source and
runtime artifacts are archived under `/home/user/epd-v2-artifacts/` on the execution
host; [artifact_locations.json](artifact_locations.json) binds paths to SHA256.
These are local retained artifacts, not publicly hosted downloads. Reproduction
from the checked-in recipe uses a new exclusive directory.

- Smoke state_dict + optimizer checkpoint SHA256:
  `fcbaf0d593804bcaefdad0e834ad90393692551bca3101755df3634011f5cd0f`
- Explicit torchvision COCO_V1 initial weights SHA256:
  `bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0`
- Unchanged baseline ONNX SHA256:
  `21362d62d816bd684f2b5c7769d0bcd1cd568af86419788fca9d095c46f2bad2`
- **V2 ONNX reference/hash: absent — not exported.**

[scenes.json](scenes.json) freezes all scene seeds and membership before training.
[acquisition.json](acquisition.json) contains per-scene raw raster/native report/
world/spec/log hashes and genuine process outcomes. [training_run_config.json](training_run_config.json)
records the exact executed source/config, versions and pretrained initialization.
The archive includes source snapshots from each executed stage; later code-review
fixes to split rejection, validation false-positive scoring, memory sampling and
diagnostic wrappers are not presented as the originally executed acquisition code.
These fixes did not rerun acquisition or training or alter their outputs.

Both failed and corrected diagnostic process results and compressed stdout/stderr
are checked in. Baseline ORT emitted internal shape-annotation warnings; output
arrays passed actual N/4, N, N and N/1/512/512 checks. Native Gazebo logs include
an SDF conversion warning and an EGL software-selection warning; capture reports,
GL attachment checks and process exits passed. The native profile makes no claim
of stock RGB equivalence. No warning or failed attempt is silently replaced.

The initial unit-test timeout fixture also failed because its Python subprocess
did not print before a 0.3 s timeout. A shell fixture with 1 s timeout now verifies
that timeouts preserve logs. Final verification: **8 focused unit tests pass**,
Python compilation passes, native compilation passed, and offline metrics replay
matches the stored genuine outputs. Full ROS integration and GPU reproducibility
were not tested.

## Historical failure and remaining qualification

The historical two-cube input is excluded from training and selection. Its
[retained native reference](https://github.com/Mukaram01/Easy_Manipulator_Improved/blob/e25d03831292098543c2ed3467d50c65ce000762/evidence/stage_a2_geometry/physics_owner/same_fragment/paired_native_reference/README.md)
records baseline raw native spill **456 / 635 px**, IoU **0.939912 / 0.930768**,
boundary means **1.440460 / 1.662152 px**, zero cross-object foreground and zero
strict accepted associations. These are existing baseline evidence, not new v2
measurements. A compatible v2 export does not exist, so the v2 regression is
explicitly **NOT RUN**.

EPD main still expects BGR-mean input and box-relative mask handling, whereas
the historical instrumentation and torchvision use RGB01/full-image masks.
The adapter is a concrete compatibility prerequisite, not something an ONNX
filename establishes. Export, actual EPD consumption, strict conservative
association and a one-time frozen held-out evaluation are **not implemented or
claimed complete**. Both strict association counts above remain unavailable;
native pixel-purity rejection counts do not substitute for them.

**Next single action:** implement and verify an explicit torchvision-to-EPD
input/output adapter against actual P3OrtBase preprocessing and mask reconstruction.
After that, a CUDA host is required for the bounded full-training attempt; this
CPU-only run deliberately stops at smoke. Do not spend the frozen test split until
selection and export compatibility pass.

Progress towards the full-scale dataset is **24/300 train, 12/100 validation,
12/100 held-out acquired**; remaining acquisition **276 / 88 / 88**. Scaling must
broaden geometry, lighting/material/background coverage and independent seeds,
then satisfy the strict segmentation and unchanged fail-closed association gates.
The pilot is **not a successful end-to-end engineering milestone**, because full
training/export/held-out evaluation have not worked yet. Conservative geometry,
the **0.1 mm physical-contact requirement**, EMD PR #3176 and deployment behavior
are untouched. No MoveIt, controller, physical-motion or Workcell Studio action ran.
