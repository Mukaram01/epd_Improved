# Independently admitted four-scene frozen-logit shadow experiment

**Completed once, diagnostic only.** Explicit user Option1 authorisation permits
reuse of the exact56-grid checkpoint solely for this fixed experiment. Original
training parent exit, wall and resource enforcement remain **UNKNOWN**. The old
strict admission code/preregistration are unchanged; Stage-A2 stays **BLOCKED0/2**.
No checkpoint is promoted and no production paste code is replaced.

## Separate admission and independent review

New `diagnostic_shadow_capture.py` validates exact checkpoint/dataset/scenes,
fixed three arms, purpose, limits and output path. It requires a SHA-pinned
review of the exact qualification, sources and evidence by a different agent;
missing/rejected/self-issued/drifted review blocks before model loading.
Qualification is one-use through the exclusive fixed output directory. Review
is authenticated by this session's separate-agent execution and retained record,
not a third-party cryptographic signature; it is not a general exception API.

Independent reviewer **`/root/diagnostic_admission_review`** approved before
inference, after independently checking25 source hashes,78 evidence hashes,
four scene provenance bindings,300-step history,12 source snapshots, checkpoint
bindings and295 frozen non-mask tensors/buffers. Decision:
**APPROVE_DIAGNOSTIC_ONLY**, no actionable findings. The implementer did not
approve its own exception. Qualification and review were committed before
execution in `b436ca86e486ecfb447ba456d9cedd8b12cd20f5`.

| Bound record | SHA256 |
|---|---|
| Qualification | `02c3d52cb3e846b30224c5a6d37b34ef3ad7a5236d5229fe5cbb3c4e25a1f4f6` |
| Independent pre-inference review | `63b4512bcbd1af685e5d7a4407accde96dd730a5dc28478a1065038a3d3b1545` |
| Executed capture source | `12702816829118cb8bac72bda2ec17e3b2e4772bc5a73351b49ab42cc273177f` |
| Original unchanged preregistration | `dada1f4af5c18e3fa7ce7f214c6e0adb4e2c2a24403dd9b49980d63554890d75` |
| Frozen checkpoint | `bf3ac58116a3a692806ed20b25fb5414277e76da605cf0f60e2e5a23656bfbfe` |
| Frozen dataset manifest | `04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11` |

The complete source/version/library/input inventory is retained in qualification
and results. Fresh tensor-only integrity rechecks ran before capture, without
turning original worker publication into parent-supervisor success.

## Execution and evidence capture

Exactly one CPU model forward per preregistered scene024..027,2 Torch threads,
1 interop thread, no warmup forward. Hooks retain **actual N×2×56×56 pre-sigmoid
logits**, transformed boxes and class/score consistency; raw captures retain
original512 boxes/scores/classes/full probabilities, RGB256 and both native ID
rasters. Original→transformed image sizes and per-array SHA256 are recorded.
All four capture/paste parity gates passed **before any shadow arm comparison**.

All weights were frozen, model eval-only, deterministic algorithms enabled.
Weight-state digest before/after:
`f6b3a03aa35de0ff581ccb061cd697701312882a6eda9116af3bca5762ff9ed3`.
Predictions/boxes/logits remain unchanged; controls reconstruct transient shadow
probabilities only. No corrected mask is exported for production use.

| Scene | Candidate masks checked | Confident detections | Visible cube targets | Paste max abs / threshold differences | Historical boxes/labels/scores/masks |
|---|---:|---:|---:|---|---|
| scene_024, negative | 72 | 0 | 0 | **0 /0**, bitwise | All bitwise equal |
| scene_025 | 2 | 1 | 2 | **0 /0**, bitwise | All bitwise equal |
| scene_026 | 13 | 3 | 3 | **0 /0**, bitwise | All bitwise equal |
| scene_027 | 1 | 1 | 1 | **0 /0**, bitwise | All bitwise equal |
| **Total** | **88** | **5** | **6** | **Exact** | **Exact** |

The negative scene retains72 low-confidence candidates for parity, but zero
confident detections/false positives in every arm; mean target IoU is null, not
an invented background-target IoU. Generic zero-candidate captures are also
tested. Baseline matching is frozen across arms. The unmatched scene025 cube
ID65503 remains in every arm with318 missing native512 pixels and80 missing256
pixels, IoU0. No control can recover a missing proposal under frozen scores.
Five of six instances meet the diagnostic native IoU>=.5 count in every arm;
that is **not EMD strict association or Stage-A2 acceptance**.

Raw renderer labels include nonzero class0 table/wall IDs. For **scoring only**,
those labels are interpreted as background0 using the retained class mapping;
other cube IDs remain nonzero cross-object truth. Raw IDs are retained intact.
This label interpretation never trims/corrects a model mask or feeds renderer
truth to the network. Native512/256 labels are independently rendered.

## Actual fixed-arm measurements

IoU is mean over all six visible instances, including the unmatched instance.
Background/cross-object counts sum all confident detections; missing counts
sum instances under the frozen baseline assignment. No matching is retuned.

| Raster / arm | Mean instance IoU | Background spill | Missing foreground | Cross-object |
|---|---:|---:|---:|---:|
| Native512 full / baseline | 0.7625050384 | 453 | 1074 | 285 |
| Native512 full / expanded-box round | 0.7651447016 | 364 | 1156 | 266 |
| Native512 full / exclusive extent | 0.7555777522 | 290 | 1370 | 272 |
| Native512 full / align_corners=True | 0.7590848659 | 571 | 1034 | 299 |
| Native512 original ROI / baseline | 0.7604984404 | 441 | 1131 | 285 |
| Native512 original ROI / expanded-box round | 0.7617048311 | 344 | 1249 | 266 |
| Native512 original ROI / exclusive extent | 0.7555777522 | 290 | 1370 | 272 |
| Native512 original ROI / align_corners=True | 0.7570415420 | 556 | 1095 | 299 |
| Conservative ANY2x2 / baseline | 0.7541832580 | 214 | 214 | 92 |
| Conservative ANY2x2 / expanded-box round | 0.7624394969 | 165 | 230 | 82 |
| Conservative ANY2x2 / exclusive extent | 0.7610296703 | 148 | 233 | 87 |
| Conservative ANY2x2 / align_corners=True | 0.7482450622 | 255 | 211 | 95 |

Per-scene original-ROI / mapped background spill, to expose aggregate tradeoffs:

| Scene | Baseline ROI/map | Round ROI/map | Exclusive ROI/map | Aligned corners ROI/map |
|---|---|---|---|---|
| scene_024 | 0 /0 | 0 /0 | 0 /0 | 0 /0 |
| scene_025 | 91 /45 | 148 /59 | 30 /18 | 121 /56 |
| scene_026 | 262 /119 | 136 /67 | 204 /94 | 307 /129 |
| scene_027 | 88 /50 | 60 /39 | 56 /36 | 128 /70 |

All per-scene/per-instance metrics, edge histograms and changes are in the
committed byte-identical `measured_results.json` and original `results.json`.

### Geometry contribution supported by the intervention

The controls isolate effects of individual **post-logit paste rules conditional
on these fixed logits/boxes**; they are not additive causal error partitions.

- **Expanded-box rounding:** native-ROI background−97 (**−22.00%**), mapped−49
  (**−22.90%**); but native-ROI missing+118, mapped missing+16. It worsens scene025
  spill and scene026 native cross-object contamination (+10). Full native IoU
  increases in aggregate, but per-scene harms invalidate a fix claim.
- **Exclusive extent:** native-full background−163 (**−35.98%**), native-ROI−151
  (**−34.24%**), mapped−66 (**−30.84%**). Full native missing+296, ROI missing+239,
  mapped missing+19. Native-full IoU drops0.006927; native-ROI IoU drops0.004921,
  despite mapped IoU improving0.006846. Reducing spill by changing extents loses
  valid foreground; mapped IoU alone would conceal that tradeoff.
- **Aligned corners:** native-ROI background+115 (**+26.08%**), mapped+41
  (**+19.16%**), cross-object+14/+3 respectively, while missing falls36/3.
  This alternative expands support and does not improve the spill problem.

All three arms **FAIL the preregistered materiality/no-harm rule**. Exclusive
extent improves both spill counts on3 positive scenes and rounding on2, but
neither preserves per-scene completeness/IoU/cross-object behavior. No winner
is selected, no arm is promoted, and no production threshold/paste changes.

### Edge distances and conservative projection

Distance bins are raster pixels <=1,(1,2],(2,3],>3 to the same-resolution visible
reference ID. They do not measure continuous geometry or physical contact.

| Arm | Native-ROI background distance bins | Native-ROI max | Mapped background distance bins | Mapped max |
|---|---|---:|---|---:|
| Baseline | 359 /68 /6 /8 | 6.708204 | 204 /7 /3 /0 | 2.236068 |
| Round | 274 /49 /11 /10 | 5.830952 | 148 /14 /3 /0 | 2.236068 |
| Exclusive | 234 /54 /2 /0 | 2.236068 | 145 /3 /0 /0 | 1.414214 |
| Aligned corners | 435 /99 /9 /13 | 7.615773 | 240 /10 /4 /1 | 3.162278 |

Baseline union foreground occupancy1/2/3/4 is **98/206/107/4330 cells**;
813 native slots inside selected cells are inactive before union. Among214
mapped-background cells, **197** have partial support, **111** have four native
background labels, and **108** of those111 have partial support. Another48 have
native target support with no active native background (not necessarily no
other-ID support). These are resolution/projection observations, not proof
that all214 are learned errors or independently removable union artifacts.

For rounding/exclusive/aligned-corners respectively, mapped-background partial
support counts are146/137/227; all-native-background cells76/78/143. Total mapped
foreground added/removed cells are48/123,22/112,48/1. Those totals include target
cells and must not be confused with background-spill deltas. The exact ANY2x2
contract is unchanged in every arm; no alternate projection is evaluated.

## Supervision, verification and outputs

Actual single run: exit**0**, no resource-limit reason, worker wall
**51.60647160199733s**, total**55.66972083099972s**, worker peak**1827708KiB**.
CPU2 threads, wall300s and RSS6GiB checked. Logs and launch/worker identity,
qualification, captures and result records are exclusively created and fsynced.
SIGKILL parent-death signalling closes the original detached-worker gap;
`finally` kills the worker process group. VmRSS/VmHWM are sampled every20ms and
`wait4` high-water is also checked. This is **polling plus post-exit rejection**,
not a kernel `memory.max` guarantee against transient overshoot. No limit was
approached or triggered in this run. Resource-failure tests retain logs and
block completion. No automatic retry path exists.

**50/50 focused tests PASS**,0 failures/errors, unittest1.041s; whole process
4.41s, peak619500KiB. Eight new regressions cover diagnostic scoping, independent
review enforcement, capture/identity corruption, semantic background, negative/
unmatched handling, frozen assignment, exclusive failure retention and actual
dummy wall/RSS terminations. Existing tests cover exact paste/subpixel parity,
immutable logits/boxes, history/checkpoint/frozen-state bindings and original
strict gate. No broad training or model-experiment suite was run.

The independent reviewer also verified postrun captured array hashes, historical
prediction identity, baseline parity, all existing-arm metrics, frozen matching,
unmatched targets, class0 scoring, aggregates/materiality and resource evidence,
**without new inference**. No actionable findings. Postrun review SHA256:
`6ad8d195775abdd7f59355cb63c05d53c595502b76394b644fa8d72e04414af6`.

Qualification/review root:
`/home/user/epd-v2-artifacts/mask-paste-diagnostic-qualification-001`.
Single experiment root:
`/home/user/epd-v2-artifacts/mask-paste-shadow-experiment-001`.

| Output | SHA256 |
|---|---|
| `results.json` | `f78a92515d332d89c3234ab08485e8bbf4325a2bcc928caf0d711c1a2abe8b0c` |
| `completion.json` | `ca4894d82d1881797d1b253a53c2196c6516abe225dcee21716f3da88e522ab4` |
| `supervision.json` | `a45527360bcc88c5a789c74f68d92015ea749c322f0fa828167fea63e2f935ab` |
| `scene_024_capture.npz` | `7d00f0ca5ff826f1d4ab147a7be8165c38eaf21278ffec62de93ce2529158c07` |
| `scene_025_capture.npz` | `5b2d9b1ae6c71c861938ba09a04a5238dde2ba71284a4d5a99cd140517ca3556` |
| `scene_026_capture.npz` | `df3c24c878c7b75d8991436b77e2ceeaa9cbf3002653c17575eb0f325d45ce7c` |
| `scene_027_capture.npz` | `b49b75a0a9ee6fabbf2130d62905724d8ea38a5153c59cf8b010d0490a8604bc` |

Captures remain external. Qualification, pre/post independent reviews, measured
JSON results, completion, supervision and focused test log are retained in
`evidence/segmentation_v2/shadow_diagnostic_qualification_001` byte-for-byte.
Prior artifacts, branches and original EPD/EMD checkout are preserved. No
training, held-out/Stage-A2 acceptance access, threshold change, ID trimming,
erosion/dilation, export, ROS/Gazebo, deployment or motion occurred.

Historical single command (output already exists, **do not rerun**):

```bash
python3 tools/segmentation_v2/diagnostic_shadow_capture.py \
  --qualification /home/user/epd-v2-artifacts/mask-paste-diagnostic-qualification-001/qualification.json \
  --review /home/user/epd-v2-artifacts/mask-paste-diagnostic-qualification-001/independent_review.json \
  --review-sha256 63b4512bcbd1af685e5d7a4407accde96dd730a5dc28478a1065038a3d3b1545
```

## Conclusion and one next engineering milestone

Post-paste geometry demonstrably changes contamination with identical logits,
boxes and scores, so another epoch cannot be assumed necessary. However no
preregistered geometry alternative improves spill without forbidden harms;
exclusive extents still leave290 native-full background pixels and148 mapped
cells. Those residuals cannot be uniquely assigned to learned logits, ROIAlign,
box localization or renderer rasterization. Four validation scenes do not
establish generalization, continuous silhouette truth or physical safety.

**Next milestone: a synthetic ROIAlign→paste round-trip coordinate-contract
benchmark with analytic silhouettes and subpixel boxes, before any paste patch
or retraining.** The observed spill/completeness tradeoff makes a coordinate-
consistent reference more informative than selecting an arbitrary integer or
interpolation toggle. That benchmark would isolate upstream/downstream grid
agreement without tuning on Stage-A2; it is recommended, not implemented here.
Production code stays unchanged. Original training supervision **UNKNOWN**;
Stage-A2 **BLOCKED0/2**; no promotion.
