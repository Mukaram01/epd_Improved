# Read-only native Stage-A2 boundary audit: 56-grid vs #221/#222

**Code-only, diagnostic scope. NOT executed on Ubuntu yet. Keep the PR draft.**

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

After tests PASS, execute this offline-only audit **once**, using a
**new nonexistent** output directory:

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

**Next decision:** inspect native/mapped spill distance distributions,
the exact pixel-overlap deltas and high-probability spill fractions.
Then preregister *one* hypothesis-driven training/data/geometry test
only if the diagnostic distinguishes a meaningful failure mode. Do not
conduct a sweep, mask erosion, threshold relaxation or automatic retry.
