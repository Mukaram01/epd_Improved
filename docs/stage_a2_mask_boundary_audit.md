# Read-only native Stage-A2 boundary audit: 56-grid vs #221/#222

**Ubuntu execution completed: 12/12 focused tests PASS and one retained-input audit completed. Diagnostic only; keep PR draft.**

## Why this is necessary

The one-epoch 56-grid mask candidate passed frozen validation (18/22 recall,
IoU 0.739693, spill 1238, one false positive), but failed strict Stage-A2
identity association (0/2). The two confident cubes' actual 56-grid
post-paste probabilities have native 512 background counts **194/242**
and unchanged ROI counts **194/242**; conservative256 IDs report
**103/64** background pixels, with mapped missing **0/0**.
That is an improvement relative to #221 mapped **154/160** and #222
**144/157**; neither previous candidate passed 2/2. The current
validation IoU nonetheless worsens relative to #221 (0.758659).

The 56-grid checkpoint is SHA-256
`bf3ac58116a3a692806ed20b25fb5414277e76da605cf0f60e2e5a23656bfbfe`.
The completed campaign has a hash-verified checkpoint, 300 unique recorded
training updates and an eligible validation result, but its parent
`process.json` is **missing**. Do not reconstruct/synthesize it or
label parent supervision exit0.

## Run exclusively on the Ubuntu system holding the originals

Use the **independent** EPD worktree `/home/user/epd-v2-supervised-pilot`;
do not touch the dirty original ROS workspace checkout. First run fast tests:

```bash
cd /home/user/epd-v2-supervised-pilot
python3 -m unittest discover -s tools/segmentation_v2 \
  -p 'test_stage_a2_boundary_audit.py' -v
```

The following is the **historical completed command**, not a rerun instruction. The output directory already exists; do not repeat this invocation or overwrite its evidence:

```bash
python3 tools/segmentation_v2/stage_a2_boundary_audit.py \
  --stage-a2 /home/user/epd-v2-artifacts/mask-resolution-stage-a2-001 \
  --campaign /home/user/epd-v2-artifacts/mask-resolution-campaign-001 \
  --emd-scripts /home/user/workcell_ws_stage_a2/scripts \
  --output /home/user/epd-v2-artifacts/mask-resolution-boundary-audit-001
```

No model is loaded into PyTorch. No training, inference, Gazebo, ROS, camera,
MoveIt, collision geometry edit, mask change, EMD call or motion takes place.

Outputs (only in the **new output** directory):

- `boundary_report.json`: hash-bound source inventory; exact raw vs
  ROI-cropped native512 counts, conservative mapped256 ID inventory and
  distance-to-visible-ID silhouette; spill probability >0.9 counts; missing
  and cross-object measurements; explicit native-versus-repeated256
  disagreement; #221/#222 source-backed comparisons; stage failure and
  missing supervisor provenance; explicit **UNKNOWN causal attribution**.
- `boundary_coordinates.npz`: exact `(y,x)` coordinates of the native512
  background detections and mapped256 background cells for each of the
  two confident cubes, plus native post-paste probability and nearest
  visible-object distance for each native pixel. These arrays are
  **diagnostic evidence only** and are never used to correct a mask.

Distances are measured at the **native renderer raster resolution**, using
OpenCV exact Euclidean distanceTransform to the visible object ID cells.
A corresponding distance diagnostic is reported at original256 ID
resolution. They represent raster-space distances, **not** metres, 3D
contact, penetration, continuous silhouette truth or calibration.

The audit copies neither the pixels nor checkpoint into the repo. It
verifies the selected candidate SHA, recorded training order/validation,
current Stage-A2 strict failure, source hashes of the same checker and EMD
association code, all retained capture SHA256 entries, and independent
native256/native512 identity bytes. It recomputes every confident
detection's exact earlier Stage-A2 background and mapped-ID counts.
Any unexpected mismatch **BLOCKS** the report instead of silently
updating the acceptance data.

## Earlier model comparisons

Committed `evidence/segmentation_v2/mask_finetune_001` (#221) and
`evidence/segmentation_v2/boundary_mask_001` (#222) contain previous
read-only strict Stage-A2 reports. The audit requires both reports to match
their own committed `artifact_locations.json` SHA-256 and the **same**
capture RGB/manifest identities. Where matching external
`predictions.npz` archives exist, it additionally verifies their pinned
SHA-256 and records spatial **shared/removed/added** spill, separately
for native512 and mapped256. This is a **comparison of already recorded
model outputs**, not an erosion or prospective corrected mask.
The older #221 strict report has a **legacy schema**: it contains
`mapped_id_counts` for both confident detections, but does **not** contain
`reference_identity`, `mapped_missing_pixels` or `native512_raw_missing`.
For #221 only, the audit derives an identity **solely if** that archived
detection has exactly one nonzero renderer ID whose value matches the
explicit ID of the same cube in the later report. It records
`DERIVED_FROM_UNAMBIGUOUS_RECORDED_ID_COUNTS`; ambiguous/mismatched ID
counts fail closed. Unrecorded fields remain `null` and are explicitly
labelled unknown. If the hash-verified old prediction archive is present,
its recomputed metrics may be compared separately, without pretending
the old JSON recorded them.

If a prior predictions archive is absent, it reports
`pixelwise_overlap: UNAVAILABLE`, retaining the hash-verified metric
summary only; it does not invent an overlap value. A present but tampered
archive is a hard failure.

## Limits and decision gate

The full-image Torchvision probabilities already include model mask-head,
ROI boxes/paste, and interpolation. They are **not raw 56-grid logits**.
ROI clipping and 2x2 conservative mapping are measured; no exclusive
causal split between learned weights, box placement, mask paste,
interpolation or renderer raster truth is possible without separately
controlled observations. The native512 renderer ID image is **not**
a doubled version of the original256 ID image; differences are measured,
not treated as model errors.

Result status is always `DIAGNOSTIC_ONLY_BLOCKED_NO_PROMOTION` even if
the audit itself successfully completes. The unchanged strict association
remains **0/2**, the parent supervisor record remains missing, the model
is not promoted/exported and physical grasp is unqualified.

## Completed Ubuntu boundary audit: observed results (2026-10-14)

Operator ran `python3 -m unittest discover -s tools/segmentation_v2 -p 'test_stage_a2_boundary_audit.py' -v` on commit `1558a7f`. **12/12 PASS**, 0.347s. One previously failing synthetic test was fixed with an unambiguous ID fixture; no model/artifact changed.

The single offline audit already generated:
`/home/user/epd-v2-artifacts/mask-resolution-boundary-audit-001/boundary_report.json`
and its retained `boundary_coordinates.npz`. Its reported decision is
`DIAGNOSTIC_ONLY_BLOCKED_NO_PROMOTION`; strict Stage-A2 **0/2**;
`training_supervisor_evidence=MISSING_UNRESOLVED`.
Historical archives #221/#222 both have `pixelwise_overlap=VERIFIED_ARCHIVED_RAW_PREDICTIONS` using their committed SHA256 manifests.

**Current 56-grid two confident detections (native512 distances are to independently retained visible renderer-ID silhouette):**

| Metric | Cube1 ID65517 | Cube2 ID65523 |
|---|---:|---:|
| Raw native512 background | 194 | 242 |
| ROI native512 background | 194 | 242 |
| Mapped256 background | 103 | 64 |
| Native spill distance <=1 px | 176 | 175 |
| Native distance (1,2] px | 18 | 64 |
| Native distance (2,3] px | 0 | 3 |
| Native spill >3 px | 0 | 0 |
| Native maximum distance px | 2 | 3 |
| Mapped spill distance <=1 px | 103 | 62 |
| Mapped spill distance (1,2] px | 0 | 2 |
| Native foreground spill probability >0.9 | 0 | 13 |
| Median native spill probability | 0.5932828188 | 0.6960796118 |
| Native spill removed by bbox | 0 | 0 |
| Native512 vs repeated256 ID disagreement within ROI foreground | 101 | 173 |

All **165/167 mapped false-background cells** are within 1 pixel of the object ID silhouette at 256; the remaining 2 at ~1.414px. At 512, 351/436 false-background pixels are within 1px, 82 are in (1,2]px and 3 in (2,3]px. Those measurements are raster distances, not continuous geometric error. The 101/173 ID disagreements count **native512 versus repeated native256 ID labels under foreground**, not unique causes of background spill; do not subtract them or call them registration errors.

**Exact hash-verified spill overlap (old versus 56-grid, each confident cube):**

| Comparison | Cube | Mapped256 shared | Mapped256 removed | Mapped256 added | Native512 shared | Native512 removed | Native512 added |
|---|---:|---:|---:|---:|---:|---:|---:|
| #221 | 1 | 84 | 70 | 19 | 146 | 164 | 48 |
| #221 | 2 | 63 | 97 | 1 | 230 | 296 | 12 |
| #222 | 1 | 84 | 60 | 19 | 144 | 126 | 50 |
| #222 | 2 | 63 | 94 | 1 | 233 | 269 | 9 |

Relative to #221, 56-grid reduced mapped spill from 314 to 167 (net -147 = 167 removed - 20 added). Relative to #222, mapped spill changed from 301 to 167 (net -134 = 154 removed - 20 added). The 56-grid gains are real on this scene but not an overall validation-IoU gain. Native bbox clipping removed zero background pixels in both new cubes. **Persistent near-edge false background plus 20 new mapped false cells** means increased mask resolution alone did not solve strict identity association.

### Interpretation and bounded next hypothesis

**Observed:** every mapped spill pixel lies within ~1.414 px of its same-resolution object silhouette; the unresolved failure is strongly *edge-localized*. No observed far-field blob. Even the 56-grid native spill pixels are <=3 px from visible-ID boundary. There is substantial independent 512-vs-256 ID disagreement within predicted foreground and no current bbox-cropping benefit.

**Not established:** which share is caused by learned mask logits, Torchvision mask paste, floating-point box alignment, conservative 2x2 union mapping or independently rendered resolution-specific rasterization. Neither ID raster is model input or proof of continuous 3D geometry. Do not choose erosion, trim by renderer truth, relax `>0.5`/`>=0.8`, or call the model safe for grasp.

**Smallest next step for Codex: a new read-only retained-artifact 2x2 boundary-cell attribution diagnostic, not training.** On each **mapped256** background cell where the current 56-grid mapped mask is foreground, categorise the four corresponding **native512** renderer ID samples by counts of target ID / other nonzero ID / background; count how many of those four native512 model probabilities exceed 0.5, and distinguish 256-background cells that have any native512 same-object ID support from cells whose native512 labels are all background. Count score-margin buckets above0.5 and compare the already-retained #221/#222 predictions when SHA bound. Ensure mutually exclusive buckets sum to the original 103/64 and are limited to diagnostic reporting with exact source/capture hashes. This may distinguish renderer raster disagreement and conservative 2x2 union sensitivity; it cannot uniquely identify learned-mask/paste causes. Require targeted synthetic tests, no re-inference, and a new exclusive output folder if files are generated.

**No further training experiment authorised.** Resolve parent training supervisor provenance separately; do not fabricate a missing `process.json`. Keep PR draft and EMD strict association blocked 0/2.


## Completed single 2x2 retained-cell diagnostic (2026-10-14)

Clean branch safely fast-forwarded from `1558a7f` to `5f7e325` before work.
`stage_a2_boundary_cells.py` implements the authorised bounded follow-up.
It SHA-pins the completed report, verifies its coordinate archive, recomputes
all retained current measurements/coordinates, verifies original capture bytes
and prediction/process/result hashes, and checks both historical identities
and archived metrics. Historical manifest hashes are independently pinned to
committed base `5f7e325`; present unpinned/corrupt archives block, absent ones
remain unavailable. It never loads a model or writes an acceptance mask.

Actual command, executed **once**, exit **0** (historical; do not rerun):

```bash
python3 tools/segmentation_v2/stage_a2_boundary_cells.py \
  --output /home/user/epd-v2-artifacts/mask-resolution-boundary-cells-001
```

All output remains exclusively in that new directory:

- `boundary_cells_report.json`: SHA256 `e651ac21fb4eb3d4aa5056a79ee29ac1e9a6fb86e97db7b12f04c409afac98b3`
- `boundary_cells.npz`: SHA256 `2407e96f1d7195d430bd1e142f4028699352a8877b35a6c1f926180e2fa8a409`

The input retained report SHA256 is
`4b5a82edf7169a64940f4b410e5680a2c4662dba7d29282006165c3b8a8630d9`.
The NPZ retains `(y,x)`, four native uint32 IDs, four **unmodified float32**
probabilities, ROI membership, strict-threshold activity and category indices.
Sample order is top-left, top-right, bottom-left, bottom-right. Historical
samples include each archive's own spill cells and its original probabilities,
ROI/activity at every current spill cell, including historical negative cells.

### Exclusive cell categories

Cross native target/other ID presence with any active native background sample.
An active sample is inside the original integer-truncated ROI and has `p>0.5`.
The target-support rows below have no other ID support. Every other-ID category
is zero; all-background/no-active-background is impossible for a selected cell.

| Category (cells, not native pixels) | Cube1 ID65517 | Cube2 ID65523 |
|---|---:|---:|
| Native target support; no active background | 24 | 15 |
| Native target support AND active background | 9 | 4 |
| All four native IDs background; active background | 70 | 45 |
| Any other ID support | 0 | 0 |
| **Conserved mapped256 spill total** | **103** | **64** |

Thus **39/167** cells are resolution-raster disagreements with exclusively
target-labelled active samples. Another **13/167** combine native target
support with actual native background foreground. The remaining **115/167**
have no native target/other ID support at all. These are observational
partitions, **not exclusive learned/paste/raster causal labels**.

| Active samples per 2x2 cell | Cube1 | Cube2 |
|---|---:|---:|
| 1 | 18 | 7 |
| 2 | 73 | 47 |
| 3 | 3 | 4 |
| 4 | 9 | 6 |
| Inactive slots selected by union, sum(4-k) | 203 | 119 |

**152/167** mapped spill cells have partial support (k<4). Within the
all-background category, **68/70** and **41/45** have partial support; **2/4**
respectively have four active background samples. Union therefore selects
109 all-background cells with partial support and 6 with full support. The
322 inactive slots quantify support expansion; they are neither 322 extra
mapped spill cells nor a uniquely removable causal contribution. No alternate
mapping or correction is applied.

Inside these mapped-background cells there are 209/137 active native samples:
target **58/34**, background **151/103**, other **0/0**. All four ID samples
sum to target **64/35**, background **348/221**, other **0/0**, total **412/256**.
This cell-restricted diagnostic is not a partition of the entire native spill
194/242: **43/139** native background spill pixels lie outside mapped256
background cells. In particular, the prior 13 native spill probabilities >0.9
are outside this selected cell population; none selected here exceeds 0.9.

### Threshold margins and archived comparisons

Margins are float64 `p-0.5` from original float32 probabilities. Histogram
intervals are left-inclusive/right-exclusive, final interval right-inclusive.
Zero-margin samples are inactive; positive tables below contain active samples
only. No threshold changes were evaluated.

| Positive margin bins | [0,.05) | [.05,.1) | [.1,.25) | [.25,.4) | [.4,.5] |
|---|---:|---:|---:|---:|---:|
| Current cube1 active samples | 69 | 54 | 68 | 18 | 0 |
| Current cube2 active samples | 33 | 31 | 64 | 9 | 0 |
| Current cube1 maximum active score per cell | 29 | 25 | 39 | 10 | 0 |
| Current cube2 maximum active score per cell | 12 | 14 | 33 | 5 | 0 |

The report also retains all four sample margin histograms (including negatives)
and category-specific maximum-margin histograms. Exact samples allow independent
background-only histograms without treating target-support scores as background.

Both archived predictions are SHA-verified on the same capture. Categories are
computed on each model's **own** false-background cell population:

| Model | Cube | Target support, no active BG | Target support + active BG | All native BG | Total |
|---|---:|---:|---:|---:|---:|
| #221 | 1 | 50 | 11 | 93 | 154 |
| #221 | 2 | 7 | 12 | 141 | 160 |
| #222 | 1 | 52 | 10 | 82 | 144 |
| #222 | 2 | 8 | 11 | 138 | 157 |
| Current | 1 | 24 | 9 | 70 | 103 |
| Current | 2 | 15 | 4 | 45 | 64 |

All current target-support cells (24+9 / 15+4) are shared with **both** old
models. All **19/1 added** cells versus either old model are all-native-background.
For #221, shared current all-background counts are **51/44**; #222 also **51/44**.
This does not establish the cause of the new probabilities or imply monotonic
improvement in every category (cube2 target-only spill increases).

### Verification and evidence limits

Final focused command:
`python3 -m unittest discover -s tools/segmentation_v2 -p 'test_stage_a2_boundary*.py' -v`:
**22/22 PASS**, 0 failures/errors, **0.206s**. This includes 12 retained-audit
regressions plus 10 new tests for conservation, ROI alignment, strict threshold,
margin bins, identity/corrupt arrays, coordinate drift, archive absence/tamper,
manifest repinning, synthetic report publication/hash gates and output exclusivity.
Broader model/training/inference suites were not run under this task's constraints.

The one retained-data execution used diagnostic source SHA256
`a0e03062cbbfd15999f6a39c8046a5813979ac6c22e7cc5b8328b9f18a8bd1ed`.
Review then added an independent historical-manifest pin check (no numerical
logic change). The final committed source therefore differs from the executed
source. Rather than rerun or overwrite evidence, a separate read-only check
confirmed that **both manifest and archived-prediction hashes recorded by the
actual run match the new trusted pins**. The retained report is unchanged;
do not describe it as an execution of the later hardening revision.

**Smallest follow-up hypothesis:** partial edge support in the post-paste mask
contributes materially to conservative-union spill even where both renderer
resolutions agree on background (109 cells), so native-raster disagreement
alone cannot explain the residual. A single frozen-logit/frozen-box mask-paste
alignment control would be the narrowest prospective causal test; raw ROI logits
are not retained here, and this hypothesis remains unproved. **No such control,
new inference or model experiment is implemented or authorised.** No threshold
relaxation, erosion, corrected mask, held-out access, export/promotion, ROS,
Gazebo or EMD motion occurred. Stage-A2 remains **0/2** and training parent
`process.json` remains **MISSING_UNRESOLVED**. Keep PR #225 draft.
