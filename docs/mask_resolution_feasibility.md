# 56×56 learned ROI masks: GO for one future epoch, accuracy unproven

Base #222 (`21121f0`). This is an isolated CPU feasibility experiment, not a trained candidate, deployment or qualification. Production construction/postprocessing and original/#221 checkpoints remain unchanged. No validation, Stage-A2 or held-out inference was performed. Four disposable optimizer updates were made; no campaign or checkpoint was published.

## Why this one change

Reuse #219–#222 and retained Stage-A2 diagnostics: learned high foreground probabilities occur over native background; bilinear/padding/box placement also contribute. #222 reduced mapped spill154/160→144/157 but still accepted0/2. Resolution is a credible contributor, not an established sole cause. Native512 Stage-A2 boxes are roughly100/105 pixels wide:28 logits sample that width at roughly3.6/3.8 pixels,56 at1.8/1.9. Additional learned spatial samples may improve boundaries; strict zero-spill acceptance is not guaranteed, particularly after conservative256 mapping.

Installed torchvision0.22 uses14×14 MultiScaleRoIAlign (sampling_ratio2), four256-channel learned3×3 mask convolutions, a learned stride2 transposed convolution, then two-class1×1 logits. Change **only ROIAlign output_size14→28**. Existing learned operations now produce56×56 logits, rather than resizing28 logits. No extra parameters or replacement heads;307 state tensors strict-load unchanged. The normal mask loss projects genuine targets at the actual logits size. The mask-head activations/convolution work grow approximately4× per ROI; backbone/RPN/detection work and model parameter storage remain unchanged.

Standard torchvision mask paste still applies one zero-pixel border, expanded boxes, bilinear paste and full-image probabilities. Its intrinsic expansion factor changes30/28→58/56 with resolution; this is inherent in the standard model path, not a fitted offset or altered postprocessor. It may affect the observed placement error in either direction. EPD then extracts exact clipped integer ROIs without a second resize. Architecture configuration must be explicit in future checkpoints: mask_roi_output_size28, mask_logits_size56; weights alone cannot distinguish28 from56 construction.

## Measured disposable probe

Initialize from best #221 epoch2 SHA256 `f3e7531c25cb20920c04283d205d7241028d7321b1ea34e2f1848415744a3b56`; freeze all except existing mask head/predictor. Same class mapping, FrozenBN epsilon0, RGB256→linear512/255/internal800 transform, FP32, batch1, two CPU threads, workers0, seed260110, SGD0.0001/momentum0.9/weight-decay0.0001, ordinary mask BCE×2. Do **not** inherit rejected #222 boundary weighting or its weights.

Reuse the four geometry-selected training benchmark scenes from #220. No prediction-based case selection or broad dataset audit. Existing manifest SHA256 `04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11`; selected training file hashes checked, held-out pixel reads denied.

| Training case | Instances | Actual logits | Step seconds |
|---|---:|---|---:|
| scene_000 |0|0×2×56×56|2.449269|
| expansion_122 |3|Not retained|Not retained; ≤17.902558|
| expansion_093 |3|47×2×56×56|6.606865|
| expansion_123 |1|21×2×56×56|4.050798|

First process failed after its second completed update because the evidence writer reopened exclusive `measurements.json`. Preserve exit1, traceback, executed source and audit. Correct the writer to distinct per-case files; execute **only the two unattempted cases and export**, from a fresh disposable copy. No completed batch repeated and no automatic retry occurred. These four updates across two copies are not a contiguous campaign. The missing second-step timing is not reconstructed: its entire17.902558s failed-process duration is the conservative upper bound used below.

Both processes total62.029693s (17.902557+44.127136), plus two tiny shape-test runs (<4s): well below600s. Peak reported RSS2,062,980KiB (1.967GiB), below6GiB. No resource kill. Ubuntu CPU VM,8 CPUs,~15.57GiB RAM,~81.59GiB free disk, CUDA unavailable; no dependency installs/upgrades. Retained #221 two-epoch wall2579.316s and #220 full-head maximum step4.888313s provide context only, not a controlled speed ratio for this mask-only variant.

## ONNX and real EPD interface

Disposable export opset17 took9.567567s; ORT session/inference2.733227s; actual pinned EPD C++ probe13.939131s. Input `images` float32[1,3,512,512], RGB01; outputs in order boxes float32[N,4], labels int64[N], scores float32[N], masks float32[N,1,512,512]. Dynamic detection metadata and fixed-raster output annotations checked against actual runtime shapes; internal ROI size is invisible to the adapter.

Trace outputs versus ORT maximum errors: boxes0.000175477 (<0.001), scores0.000001640 (<0.00001), probabilities0.000012815 (<0.0001); labels exact. This is disposable export feasibility, not candidate qualification. All four returned C++ masks preserve ORT ROI probabilities bitwise, with exact clipped boxes/classes. All scores are below0.8; the existing confidence-filtered comparator consequently has count0. The separate offline all-output audit verifies actual ROI values without changing any inference/acceptance threshold. No strict association/accuracy claim follows.

Retain expected tracer/constant-fold warnings and ORT internal Loop intermediate-shape warnings (appended mask dimension1 versus2/3/4). Final output shapes and numerical checks pass; do not hide these warnings. Empty ROI batches pass the actual head shape test. **Full exported-model zero-detection runtime is untested**, and remains a required future export-qualification check. No repeated original inference or synthetic replacement masks were used.

Disposable external ONNX: `/home/user/epd-v2-artifacts/mask-resolution-feasibility-unattempted-001/DISPOSABLE_UNQUALIFIED_56.onnx`,176309892 bytes,SHA256 `93a24a2720980630b6fc525a20cb92c6e74e43443dc824da7af8e386e42556f7`. It contains disposable benchmark updates, is not accepted and must not seed training. Large artifacts/raw tensors stay external; `evidence/segmentation_v2/mask_resolution_001/artifact_locations.json` pins both process directories, failures, sources, outputs and probe files. One focused shape/weight-preservation/empty-ROI test passes; red evidence retained.

## Exact future experiment admission and acceptance

**GO for exactly one epoch/300 updates, not two.** Start fresh from protected #221 epoch2 with explicit56-grid configuration and mask-only optimization described above; all300 training scenes once in existing seeded order. Keep original307 tensors, classes/preprocessing/FrozenBN unchanged, fresh optimizer/RNG and atomic checkpoint/history/source bindings. No boundary-weighting combination or sweep. One CPU execution ceiling10800s, aggregate18000s, RSS6GiB,5GiB disk reserve; no automatic retries. Using17.902558s as the conservative maximum train **and** validation-forward bound gives `(300+24)*17.902558*1.5+240 = 8940.643s`, projected RSS3,094,470KiB (2.951GiB). Retain unchanged1.5 factor and240s overhead. Two epochs project17319.040s and are **not admitted** under10800s; earlier failed admissions remain failed. Admission must verify these immutable identities and available reserve again, without repeating benchmarks.

Evaluate one completed checkpoint on the frozen12 validation scenes, compare retained original and #221 under identical scorer/preprocessing, retain every per-instance spill/missing/boundary/cross-object result and both negatives. Gate unchanged: recall≥18/22,IoU>0.685890,FP≤1,spill<1915,violations0,confidence≥0.8,mask>0.5. Only a validation-passing checkpoint gets one retained Stage-A2 check; require **both instances accepted** by unchanged strict association. IDs are offline truth only. On failure retain evidence and stop. Only if both gates pass, qualify actual candidate export including zero detections, numerical/foreground parity and EPD CPU consumption. No sealed held-out/historical regression, geometry/safety relaxation, EMD/Gazebo changes, deployment or motion.

**Single next action:** authorize that one300-update56-grid experiment. #221 remains the best validation-passing reference, but it and all current candidates remain Stage-A2 rejected. Higher resolution is affordable and technically compatible; improved accuracy remains a hypothesis.
