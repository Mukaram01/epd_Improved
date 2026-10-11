# EPD_V2_300_TRAIN_SCENES_READY

Baseline `34b78f6fe1c48e6e26185db7f1455cebf045b23d`, branch
`codex/epd-v2-supervised-pilot`, draft PR #219. Dataset acquisition and loader
milestone only. No model training, inference, selection, export or qualification.

## Actual acquisition

**276/276 new genuine scenes**, once each, in **six sealed batches of 46**.
Final membership **300 train / 12 validation / 12 held-out**. The original pilot
manifests and samples were not rewritten; all nine rejected checkpoint hashes
(smoke plus eight full-training epochs) were verified unchanged without loading.

| Batch | Successful captures | Batch wall seconds |
|---|---:|---:|
| 0 | 46 | 124.128 |
| 1 | 46 | 124.211 |
| 2 | 46 | 123.726 |
| 3 | 46 | 124.338 |
| 4 | 46 | 123.982 |
| 5 | 46 | 123.741 |

Total acquisition **744.139 s**. Native exits: **276 zero, 0 failures**. No
successful capture was rerun. Scene times **2.645–3.069 s**; sampled peak RSS
**625236 KiB**, child peak **634560 KiB**. Declared limits: 30 s/scene,
600 s/batch, 4000 s/campaign, 2048 MiB immediate-process RSS, 5 GiB free-disk
reserve. RSS sampling is not an aggregate descendant-process limit.
Initial available RAM was 11.86 GiB; free disk 85.41 GiB. No dependency upgrades.

The capture is the existing EPD-owned Gazebo Fortress 6.18.0/Ogre2 executable,
SHA256 `755af4a1c34ea22669d5df5145585826ed6bba69955ce0e56a3329a61c8ca4d3`.
It was reused from the preserved pilot build, not rebuilt in this milestone.
`native_build.json`, `native_headers.json`, `preflight.json` and
`runtime_libraries.json` retain build provenance, exact pinned EMD header identities,
source hashes, executable identity, package versions and 110 linked-library hashes.
Headers were checked directly against read-only EMD commit
`e25d03831292098543c2ed3467d50c65ce000762`. EMD files were not changed.

The camera remains source-fixed at `[0,-0.40,0.28]`, looking at `[0,0,0.025]`,
Z-up, vertical FOV 60 degrees, aspect 1, near/far 0.01/2 m. These ideal pinhole
parameters imply focal length 221.702503 px at256 and 443.405007 px at512; they
are source-defined render parameters, not a new physical-camera calibration.
Actual render attachments/viewports and single-pass/depth state are retained in
every native report. RGB and uint32 IDs share each native scene pass; the two
resolutions are captured with the same frozen ECM at step2 / stamp2000000 ns.
Model preprocessing remains **RGB256 → INTER_LINEAR512 → float32 CHW /255**.
Supervisory labels are genuine **native512 little-endian uint32** IDs; they are
not resized ID maps. The additional native512 RGB is retained for provenance,
not substituted for the intended RGB256 input.

## Variety and quality

IDs `expansion_000`–`expansion_275`, seeds `260111000`–`260111275`, were frozen
before capture in `scenes.json` (SHA256
`3ac531d12091932375503f287540244e4a6dd602799b61eaf6e8110c8e1e4e0d`).
Scene configuration does not read validation predictions, evaluation pixels or
the historical regression. All 324 scene IDs/seeds/definitions are distinct.
RGB256 and native512 RGB hashes are unique across all 324 records, verified using
original evaluation manifests without opening evaluation images. Historical RGB
hashes are excluded.

| Expansion coverage | Actual result |
|---|---|
| Authored target counts 0 / 1 / 2 / 3 | 46 / 77 / 77 / 76 scenes |
| Visible target counts 0 / 1 / 2 / 3 | 46 / 83 / 83 / 64 scenes |
| Native visible cubes | 441 |
| Fully occluded authored cubes | 18, retained in provenance |
| Arrangement tags | 78 separated, 77 touching, 75 depth-overlap, 46 negative |
| Multi-object arrangement subsets | 52 separated, 51 touching, 50 depth-overlap |
| Contrast groups | 92 low, 92 high, 92 mixed |
| Class-zero clutter | 115 scenes with one thin box; 161 without |
| Cube edges | 20.244–84.797 mm |
| Visible area min / p25 / median / p75 / max | 2 / 1501 / 3456 / 6227 / 11901 px |
| Ambient coefficient | 0.125434–0.817595 |
| Estimated occlusion <5% / 5–30% / ≥30% | 255 / 136 / 68 authored instances |

Arrangement tags describe authored configurations: one-cube scenes do not imply
inter-object contact. Occlusion is an explicit estimate comparing native visible
area with a camera-projected BOX silhouette raster, not amodal ground truth.
Native visible-pixel counts are authoritative. No labels or RGB were corrected.
The 2-pixel fragments and all other valid difficult samples were retained.
Fully hidden authored cubes remain in the scene/mapping provenance; as in the
original loader, only visible cubes produce supervised instance targets.

Materials use the existing opaque diffuse-colour shader with varied directional
lighting, backgrounds, position and orientation. This acquisition adds no texture,
PBR, shadow or MSAA capability and claims no stock-renderer photorealism. Per-scene
colours/light directions, pose ranges, pixel areas and occlusion estimates are in
`quality.json`; genuine RGB previews are in `rgb_preview.png` with scene IDs in
`preview_identity.json`.

All samples passed RGB byte-length/dtype/integrity checks, exact native ID raster
checks, synchronous native-pass validation, complete class/renderer/Item/visual/
link/model ownership and one-to-one matching BOX collision ownership/geometry.
Collision boxes were added only to the new static training worlds; no robot
collision or production safety geometry was modified. Unmapped IDs, duplicate
RGBs, repeated definitions and split contamination: **0**. Every attempt retains
raw-file hashes, native status, timing and ownership records under `attempts/`;
all six exact coverage seals are under `batches/`.

## Loader and verification

`dataset.json` SHA256:
`04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11`.
Its 324 records bind original and new data paths and hashes. The new metadata-first
loader validates exact frozen membership, canonical folders/origins and the pinned
acquisition/attempt hash chain before opening any sample files. Original pilot
folder loading remains supported through `train.verified_rows`; passing a composed
manifest delegates to the integrated loader. Training campaign execution and its
24-scene budgeting assumptions are not expanded or invoked by this dataset task.

The real supervised `torch.utils.data.DataLoader` smoke consumed **300 samples,
484 visible instances, 50 negatives**, with exact native-ID-derived uint8 masks
and RGB01 CHW tensors. It opened **300 training scene directories** and **0
validation/held-out directories**. A Python open-event audit hook rejects access
to any evaluation scene directory. No model was created, no inference occurred,
and optimizer steps were **0**. Process wall time **10.680 s**, child peak RSS
**528288 KiB**, exit0, within 300 s / 4096 MiB budgets. Per-sample input/target
hashes, access counts and runtime are in `loader_smoke.json` and `loader_process.json`.

**24 focused deterministic tests pass**. Coverage includes exclusive manifest
freezing, changed camera/split/ID/seed rejection, duplicate/historical RGB,
native label dtype/unknown IDs/negative labels/collision ownership, partial failed
campaigns, exact batch/acquisition coverage, split swapping, folder/hash substitution,
legacy delegation, supervised tensor contracts and unchanged pilot/campaign tests.
Python compilation and whitespace checks pass; exact commands/output are in
`verification.txt`. No repository-wide or ROS integration success is claimed.

## Failures, provenance and storage

The first preflight stopped before any scene attempt because the code assumed
`source_manifest.json` was checked into the pilot evidence directory. It is
retained externally. The failed preflight directory is preserved at
`/home/user/epd-v2-artifacts/training-expansion-001`; the corrected preflight checked
headers directly against the pinned EMD commit and froze a new exclusive directory.
No failed sample was replaced. `resolved_preflight_failure.json` records this.

Read-only review found split/path binding and exact batch seal gaps. Regression
tests first demonstrated rejection failures; both were fixed before integration
and the real loader smoke. The final additional review attempt was unavailable
because of the reviewer service usage limit; local focused verification completed.
The exact acquisition code is retained as `executed_acquisition.py` and externally
in `executed_sources/`. Post-capture audit/loader hardening is distinguished by
`source_provenance.json` and `capture_to_final_*.patch.gz`; no native capture code,
scene, RGB, ID or frozen plan changed.

All 276 native stderr logs retain the existing Gazebo SDF conversion message and
EGL software-selection warning. All native exits and attachment/label/ownership
checks passed. `native_log_summary.json` counts the messages; one exact compressed
example is retained. Original logs/reports remain hash-referenced externally.

Raw artifacts remain at `/home/user/epd-v2-artifacts/training-expansion-002`, with
integration under `integration/`. `artifact_locations.json` lists paths, SHA256s
and sizes, including all raw rasters and metadata. Raw raster storage is
**633077760 bytes (603.75 MiB)**; the retained expansion with metadata/source
snapshots is approximately **610 MiB**. These are execution-host local artifacts,
not a remote durability claim. Large raw RGB/ID files remain outside Git.
Original pilot storage and rejected checkpoints remain intact.

There is **no remaining acquisition/loader blocker**. This completes300 training
scenes, not300/100/100: validation and held-out remain12 each, leaving88 each for
future separately authorized acquisition. No validation/held-out evaluation,
historical regression access, checkpoint selection, export, EMD safety-code changes,
MoveIt, controllers, robot execution, deployment, merge or training occurred.
Conservative collision restrictions and the 0.1 mm physical-contact requirement
remain unchanged. Keep PR #219 draft and stop at this dataset milestone.
