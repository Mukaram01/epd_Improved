# Frozen-logit mask-paste shadow control: preregistration and admission

This bounded follow-up to draft #225 is **not a training experiment**. Its
question is whether legacy paste geometry materially contributes to spill when
learned logits, original boxes and scores remain frozen. The selected 56-grid
checkpoint is `bf3ac58116a3a692806ed20b25fb5414277e76da605cf0f60e2e5a23656bfbfe`.
Its training parent `process.json` remains absent. The user's explicit instruction
“Stop on drift or absent provenance” makes that an inference admission blocker,
even though the checkpoint and selected validation result are hash-bound.

Implemented scope: independent eager-paste reconstruction, pure shadow
measurements, regression tests, source inventory and bounded admission audit.
**No real-scene capture worker is implemented or executed while that gate is
closed.** This is a tested diagnostic core and a recorded blocked admission,
not a completed real-data geometry experiment. The audit CLI only checks and
reports provenance; even supplying a supervisor cannot make it run inference.

## Actual installed path audit

Installed versions: **torch 2.7.0**, **torchvision 0.22.0**, **numpy 1.26.4**,
**opencv-python 4.11.0.86**. The committed
`tools/segmentation_v2/shadow_mask_paste_preregistration.json` records complete
file SHA256 and function-source SHA256 inventories, including compiled
Torchvision `_C.so` and Torch `libtorch_cpu.so`. These identify the installed
runtime audited here; historic campaign binding records package versions and
project-source hashes, not historic Torchvision binary hashes. Thus installed
binary identity is **not proof of the historic process's binary identity**.

| Installed source | SHA256 |
|---|---|
| `models/detection/roi_heads.py` | `521f7de74c5950499e8f00f6a514612afa8d5746d8fc6d8e9bcc860b611d0c38` |
| `models/detection/transform.py` | `31ad020ef2c23259373314b76ac5dc0f1ad2a2c44b69c68ba62f93df82d79b50` |
| `models/detection/mask_rcnn.py` | `5f51904b7d78a8ecb8b820a9ecc3df1fa5b5d88c9d4705beb430a60a70751468` |
| `ops/poolers.py` | `cf36211fba683301a56310ef02f09a2fd7a683d5fbb0ce316452c4cb4cefbf9b` |
| `ops/roi_align.py` | `dced6acd2b5c935fe5b62c150725384e026394138febd645a10398a6c9666cd7` |
| Torchvision `_C.so` | `8cc7323e6ace3cd92531b2458381b00e6aa8f4f98406601f09e3c301a2952370` |
| Torch `libtorch_cpu.so` | `dc3f5537257da1435f82c858a92a4d7316dc6ae973f0e66f698ab39d2a70075c` |

Trace from retained project source and installed package source:

1. Project `train.tensors` reads RGB256 and uses OpenCV `INTER_LINEAR` to512,
   RGB float32 CHW `/255`. `recover_original.make_model` uses min_size800,
   max_size1333 and FrozenBN eps0. These are the recovered model settings,
   rather than the different generic `train.build_model` defaults.
2. `GeneralizedRCNNTransform` normalizes and resizes512 to800 using bilinear
   `align_corners=False`, `recompute_scale_factor=True`; 800 is already divisible
   by32, so this square image requires no extra batch padding. Image resize
   follows half-pixel sampling; no new resize/normalization is introduced here.
3. Box decode/NMS supplies continuous boxes in the transformed image. Mask
   ROIAlign uses those boxes and FPN feature scales, inferred/rounded to powers
   of2. `configure56` changes mask pool output14→28, keeping sampling_ratio2.
   `MultiScaleRoIAlign` calls `roi_align` without an aligned argument: default
   **aligned=False**, offset0 rather than -0.5; ROI width/height clamp to >=1.
   The fixed 2x2 samples per bin are bilinearly sampled in feature coordinates.
   CPU dispatch uses the installed compiled op; Python fallback source expresses
   the same documented coordinate convention but is not historic kernel proof.
4. Four mask-head convolutions preserve the28 grid; learned stride2 kernel2
   transposed convolution gives56, followed by ReLU and class logits1x1.
   `maskrcnn_inference` applies **sigmoid before pasting**, then selects the
   predicted class. Interpolating logits then sigmoid would be a different
   numerical operation and is not one of the controls.
5. Transform postprocess rescales boxes800→512 by floating multiplication
   with512/800, with no pixel-centre offset. `paste_masks_in_image` pads the
   sigmoid mask with one zero pixel on all sides (56→58), expands each box
   about its centre by58/56, then casts the expanded coordinates to int64
   (**truncation toward zero**, also for negative expanded edges).
6. `paste_mask_in_image` uses width=`x2-x0+1`, height=`y2-y0+1` (inclusive
   legacy endpoints), clamps each to >=1, then resizes58 to that size using
   bilinear **align_corners=False**. For output index u, the source-coordinate
   convention is `(u+0.5)*58/output_width-0.5` with edge handling from Torch.
   It clips the destination to the image and copies matching source slices.
   This eager legacy convention exists in installed source; its existence
   alone does not establish that it is erroneous or causes this model's spill.
7. EPD's retained Stage-A2 checker separately truncates/clips the **original**
   512 box, crops a half-open ROI, thresholds `>0.5`, then uses unchanged
   conservative ANY-of-four2x2 projection. Confidence remains `>=0.8`.
   Paste's expanded inclusive box and consumer's original half-open ROI are
   different operations. No altered projection contract is evaluated.

The independent reconstruction calls low-level Torch sigmoid, pad and bilinear
interpolate, **never Torchvision paste/expand helpers**. It reproduces their
arithmetic order. That validates orchestration parity, not an independent
implementation of the interpolation kernel. Actual eager paste is the parity
reference; no ONNX path or export is used.

## One preregistered shadow experiment

Machine-readable policy is frozen before admission. Four independent validation
scenes **scene_024/025/026/027**, seeds26011024..26011027, are selected by first
lexicographic IDs in the frozen dataset, independent of mask results. Based on
frozen `scene_plan` metadata, scene_024 is the negative control. No scene pixels
were used for selection. Dataset SHA256:
`04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11`.
All per-scene planned input hashes are retained in the JSON. Held-out records
are filtered before scene access; Stage-A2 acceptance RGB hash is explicitly
excluded. Reading manifest membership is not reading held-out pixels.

After genuine provenance admission in a future reviewed revision, capture would
require at most one CPU forward per scene, all frozen weights/eval mode, raw
class logits, class selection, full-precision boxes/scores, image transform
sizes and original Torchvision masks. This revision does not implement that
capture or claim its parity. Original logits cannot be recovered by inversion
of already-pasted probabilities. A synthetic/reconstructed supervisor or a new
file merely claiming exit0 is unacceptable: the policy supervisor SHA remains
null, requiring an independently reviewed authentic binding before admission.

Controls, fixed in advance and evaluated only after **bitwise baseline
probability parity** (also zero strict-threshold differences):

| Arm | Only changed paste rule |
|---|---|
| Baseline | Exact installed eager paste |
| `round_expanded_box` | Expanded-coordinate truncation→nearest-even rounding |
| `exclusive_extent` | Inclusive +1 extent and upper slice bound→exclusive +0 |
| `align_corners_true` | Bilinear align_corners=False→True |

Original boxes, logits, scores, confidence and mask threshold stay immutable.
Changing ROIAlign, features, logits, box proposals or input resampling is outside
this experiment. There is no shift sweep, erosion, ID trimming or chosen winner.

Metrics are full native512, original-ROI native512 and unchanged union256 IoU,
background spill, missing foreground, cross-object spill, native edge-distance
bins <=1/(1,2]/(2,3]/>3 pixels; added/removed threshold decisions and maximum
probability differences; union occupancy1..4; native512 versus repeated256 ID
label disagreement. A baseline-only renderer identity is fixed across arms;
invalid/background/fractional identities fail. Unmatched detections/missing
targets must be reported by a future capture evaluator rather than silently
removed. The pure core measures supplied fixed positive identities only.

Preregistered materiality: >=10% aggregate background reduction in **both**
native512 ROI and mapped256, improvement on >=2 positive scenes, no per-scene
IoU decrease or missing/cross-object increase, and no new negative-scene
confident detection. Report every arm even if inconclusive. This diagnostic
criterion neither selects a checkpoint nor authorises promotion. Maximum4
scenes,2 CPU threads,300s,6144MiB RSS,6GiB free-disk reserve; no retries.

## Synthetic implementation validation

Regression cases cover bitwise interpolation/paste parity (subpixel, clipped,
tiny, empty and native512 batch), deterministic reconstruction, immutable inputs,
three nontrivial one-rule controls, inclusive extent/off-by-one, strict `>0.5`,
odd ROI/2x2 alignment, native/mapped metrics, invalid IDs/arrays, baseline drift,
input/source hash binding, absent/unpinned provenance, blocked publication and
output exclusivity. Focused review found an identity coercion bug; it was fixed
with a failing-then-passing regression before admission.

For a deliberately synthetic constant-logit20 square with original box
[3.25,3.25,18.8,18.8] in a32x32 image, **full-image foreground counts** are:

| Baseline | Round expanded box | Exclusive extent | align_corners=True |
|---:|---:|---:|---:|
| 324 | 289 | 289 | 256 |

The baseline expands/truncates to[2,2,19,19], then uses18x18 inclusive extent.
This demonstrates that paste rules can change area with identical inputs;
it is not a real-scene spill improvement, nor evidence that a control fixes EPD.
No renderer labels, Stage-A2 masks or validation outputs were used to construct
this fixture. Consumer-ROI/mapped metrics are evaluated separately in tests.

Reproducible bounded test command (synthetic plus retained-audit regressions):

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONPATH=tools/segmentation_v2 timeout 120s /usr/bin/time \
  -f 'wall=%e seconds peak_rss=%M KiB' \
  python3 -m unittest -v test_shadow_mask_paste \
    test_stage_a2_boundary_audit test_stage_a2_boundary_cells
```

Admission/source audit command (writes only to a new nonexistent output folder;
**does not run inference**, even when provenance is present):

```bash
python3 tools/segmentation_v2/shadow_mask_paste.py \
  --output /home/user/epd-v2-artifacts/mask-paste-shadow-audit-001
```

The CLI supervises its audit child for300s/6144MiB, preserving exclusive logs
and `audit_process.json`. This genuine **audit** process record must never be
substituted for the missing **training** process record.

## Actual bounded admission result (2026-10-14)

Preregistration and executable source were committed **before admission** in
`ef6e6f2` (stacked on225 at `10d8e76`). Preregistration SHA256:
`dada1f4af5c18e3fa7ce7f214c6e0adb4e2c2a24403dd9b49980d63554890d75`.
Executed diagnostic source SHA256:
`1b2caf356f42fac0d4e5f13606e97b46f4fa9f4687f1bf9b3991ceabc54f6d46`.
The source remains unchanged after this run.

One invocation of the documented audit command produced
**BLOCKED_MISSING_TRAINING_PROVENANCE**, exit**2**, no resource-limit reason.
Missing authentic record:
`/home/user/epd-v2-artifacts/mask-resolution-campaign-001/process.json`.
No retry, alternate provenance or synthetic supervisor was attempted.

| Observed audit resource/use field | Actual value |
|---|---:|
| Supervised wall time | 1.753426524999668 seconds |
| Worker's measured audit time | 1.6087276180005574 seconds |
| Child peak RSS | 60476 KiB |
| Sampled process peak RSS | 44388 KiB |
| Checkpoints deserialized / models loaded | 0 |
| CPU model inference calls | 0 |
| Scene pixels / held-out files / Stage-A2 files opened | 0 / 0 / 0 |
| Real-scene baseline parity | **NOT MEASURED (null)** |
| Real-scene controlled comparisons | **NOT MEASURED (null)** |

All outputs are new; prior225 reports and predictions are preserved:

- `/home/user/epd-v2-artifacts/mask-paste-shadow-audit-001/admission/admission_report.json`,
  SHA256 `fc14cd8f1ee25adb8694679ce16608e388c129ed928784f53bcb461005e660f4`.
- `/home/user/epd-v2-artifacts/mask-paste-shadow-audit-001/audit_process.json`,
  SHA256 `89bac1467bb36dac4098749fedc2e14dfdee83c318b7a38eaf0e97b0a45090b8`.
- Exclusive `stdout.log`, empty `stderr.log` and retained `focused_tests.log`
  in the same directory. The admission report includes exact source/function,
  installed-library, checkpoint, dataset, campaign result/binding/history hashes
  and the policy hash. It records planned scene IDs, not measured scene results.

Final bounded focused suite: **35/35 PASS**, **0 failures/errors**, unittest
runtime**0.379s**; whole process wall**4.68s**, peak RSS**619052KiB**. This is
13 new synthetic tests plus22 existing retained-boundary regressions. Baseline
synthetic paste parity is **bitwise exact, max absolute difference0, zero >0.5
decision differences**. No broader model/training/inference suite was run.
`git diff --check` passed; focused independent review found no remaining
issues after identity-validation hardening.

### Evidence conclusion and one next milestone

The installed source and synthetic measurements demonstrate plausible mechanisms:
expanded-box integer quantization, inclusive endpoints and bilinear sample-grid
conventions can alter foreground with frozen logits. They **do not quantify any
share of EPD's real residual spill attributable to those mechanisms**. #225's
115 all-native-background cells and152 partial-support cells remain retained
observations; this run did not open/re-score the acceptance capture, compare
new models, or prove a learned-logit/paste/renderer causal split. No evidence
here justifies another epoch, a paste replacement or changed projection.

**One next corrective development milestone: provenance-admitted frozen-logit
capture and exact paste replay on the four preregistered validation scenes.**
Its prerequisite is recovery and independent binding of the genuine original
training supervisor record (or an explicitly authorised provenance disposition;
never a fabricated record). Only after that gate can a separately reviewed
capture worker be implemented/admitted and the already-preregistered controls
measured once. Until then real-scene root cause remains unknown.

No training, weight change, threshold adjustment, erosion/dilation, oracle
trimming, production replacement, ONNX export, ROS/Gazebo, EMD relaxation or
motion occurred. Stage-A2 remains **0/2**, checkpoint unpromoted, training
supervision **MISSING_UNRESOLVED**. Keep both PR225 and this stacked PR draft.

## Original-supervisor forensic follow-up (2026-10-15)

[Detailed provenance matrix and options](mask_resolution_training_provenance.md)
verify the worker artifact chain and saved frozen tensors, but recover no
independently authentic parent completion. Admission remains
`BLOCKED_MISSING_TRAINING_PROVENANCE`; this document's original preregistration
and executable admission gate are unchanged. No inference or training occurred.
