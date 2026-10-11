# EPD v2 supervised pilot (not qualified)

This implements real native labelled acquisition and a bounded supervised
Mask R-CNN training recipe. The recorded run acquired 48 scenes and performed
one two-step CPU training smoke. **The smoke model is worse than the baseline.**
There is no selected improved model, candidate ONNX export, verified EPD adapter,
strict EMD association result, or held-out model evaluation in this PR.

See [execution evidence](../../evidence/segmentation_v2/pilot_001/README.md).

## Dependencies and architecture

Use the installed Python environment, torch **2.7.0**, torchvision **0.22.0**,
NumPy **1.26.4**, OpenCV **4.11.0**, ONNX **1.17.0** and ORT **1.22.0**.
Native build uses Gazebo **6.18.0**, rendering **6.6.4**, Ogre-Next **2.2.5**,
jsoncpp **1.9.5**, CMake and a C++17 compiler. No environment upgrades occurred.

EPD's repository training configuration uses legacy `maskrcnn_benchmark` and
references a missing `e2e_mask_rcnn_R_50_FPN_1x_trimmed.pth`. The retained cube
model is inference-only ONNX; no compatible trainable cube checkpoint was found.
The explicit replacement training implementation is torchvision
`maskrcnn_resnet50_fpn`, initialized from named **COCO_V1** weights, with the
box and mask heads replaced for background=0/cube=1. Backbone weights are frozen;
RPN and both instance heads train. This is not fine-tuning the ONNX graph.

[Torchvision's versioned model documentation](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html)
describes the instance-mask targets, outputs and fixed-size ONNX support.
[Torchvision code licence](https://github.com/pytorch/vision/blob/v0.22.0/LICENSE)
is BSD-3-Clause; COCO training images retain their source licences. The explicit
download URL and full SHA256 are in `train.json`; there are no implicit downloads,
including backbone weights. This pilot makes no production licensing claim.

## Acquisition contract

`prepare` reads three native headers from EMD commit
`e25d03831292098543c2ed3467d50c65ce000762` using `git show`. It writes and hashes
copies outside Git. It neither changes that repository nor PR #3176.
The inherited typed GL clear/readback uses `RGBA8_UNORM` + `R32_UINT`, one
depth-tested MRT scene pass per resolution, with MSAA and blending disabled.

`capture.cpp` owns a disposable Gazebo Server and RenderUtil. Static BOX visuals
join by actual ECM visual/link/model entities to native SceneManager visuals and
Ogre Items. Every native Item must have one authoritative mapping; unexpected or
duplicate Items reject capture. `renderer_id` comes from the live visual, not a
class label or name-derived numeric identity. Authored model names bind the frozen
specification's class to that verified ECM model. Background visuals are also
mapped, with class 0. The RGB expression never uses the ID uniform.

The EPD-owned shader uses actual geometry normals, world transforms, coloured
diffuse materials, ambient intensity and a directional light. This is a bounded
renderer profile: no textures, shadows, stock-HLMS RGB equivalence, photorealism
or physical/contact truth is claimed. It extends the baseline's two-cube unlit
profile without weakening that baseline's validators.

The 48 scenes vary seed, positions, rotations, projected sizes, light direction,
ambient strength, object/background colours and depth-tested overlap. There are
8 negative images: 4 train, 2 validation, 2 test. Scene and seed membership is
frozen **before capture/training**, checked for uniqueness and exact membership.
Both historical RGB hashes are explicitly excluded; historical geometry/inputs
are never used to choose a checkpoint.

At step 2 / 2,000,000 ns, synchronous PostUpdate captures RGB/IDs at 256 then
RGB/IDs at 512 without an intervening scene/physics update. The 512 labels are
native, never enlarged 256 IDs. Retained reports include the complete ECM/native
mapping, actual attachment formats/viewports, acquisition stamp and native process
status. RGB256 is linearly resized to512 and converted to float32 RGB CHW /255,
matching the **historical** model contract. Ground-truth boxes/masks are derived
from native visible IDs for supervision only; the model takes RGB alone.

One process per scene, 30 s timeout, 900 s total, 3 GiB disk reserve; stop at first
failure, with no retry or fabricated fallback. Updated runners additionally sample
the immediate process's RSS every 250 ms and kill its process group above 8 GiB.
This is not an aggregate process-group memory limit. Original acquisition ran
before that guard was added; its measured child peak was below 1 GiB. Exact
executed sources are archived separately from later validation fixes.

## Reproduce in a new external artifact directory

Run from the repository root. `EMD_REPO` is an existing local Git repository that
contains the pinned evidence commit; it is read-only. Every acquisition and
training output directory is exclusive, so failure evidence cannot be overwritten.

```bash
python3 -m unittest discover -s tools/segmentation_v2 -p test_pilot.py
python3 tools/segmentation_v2/pilot.py prepare --emd-repo /path/to/EMD --artifacts /external/new-pilot
python3 tools/segmentation_v2/pilot.py acquire --artifacts /external/new-pilot
python3 tools/segmentation_v2/train.py fetch-weights --weights /external/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
python3 tools/segmentation_v2/train.py train --artifacts /external/new-pilot --weights /external/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth --output /external/new-pilot/training
python3 tools/segmentation_v2/evaluate_validation.py --artifacts /external/new-pilot --baseline /path/to/cube_maskrcnn.onnx --weights /external/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth --output /external/new-pilot/validation-diagnostic-corrected
python3 tools/segmentation_v2/analyze_validation.py --artifacts /external/new-pilot --output /external/new-pilot/validation-report.json
```

The last two commands are CPU-smoke validation diagnostics; they refuse a full
training checkpoint and never inspect held-out predictions. The original
`train.py` entry point implements the smoke fallback. Its full CUDA recipe uses at most eight epochs/1800 s, fixed RNG seeds,
deterministic algorithms and CUBLAS configuration. CPU performs exactly at most
two optimizer steps/300 s, saving an explicitly ineligible smoke checkpoint.
Each step saves the actual mask/class/RPN losses; CSV, JSONL and SVG retain curves.
Full training selects only on validation IoU, symmetric boundary distance and
false-positive penalty. Missed objects receive IoU=0 and boundary=512 pixels.

## Original pilot export limitation (superseded by explicit adapter)

Historical EPD instrumentation supports RGB01 and full-image-mask flags that
are absent from EPD main. Main subtracts BGR means and later resizes the entire
raw mask into its predicted box. Therefore a plain torchvision ONNX export would
be incompatible even though its output names look correct. **No export command
or held-out evaluation is advertised as implemented.** A verified explicit
input/output adapter and a fully trained candidate are prerequisites.

The diagnostic runs baseline ONNX in ORT and the smoke state_dict in PyTorch on
the same validation inputs. All outputs are genuine; masks remain `float32 >0.5`,
confidence stays >=0.8. There is no renderer-ID cropping, erosion, oracle box
adjustment or replacement detection. Erosion in `mask_metrics` extracts a
boundary for distance measurement only; it never changes a prediction.
Instance matching is one-to-one by descending IoU; recall requires IoU>=0.5.
All confident detections, including unmatched and negative-image detections,
are also scored separately. Per-detection pixel sums can count the same object
more than once; per-visible-object misses are reported separately.

Native512 pixel-purity counts are diagnostics, **not** the unchanged EMD
conservative association contract. Authoritative strict acceptance/rejection
counts remain null. The original historical regression is retained as read-only
baseline evidence, not rerun using a nonexistent v2 export. Held-out scenes remain
frozen and unevaluated until selection and actual EPD-path compatibility pass.

No geometry, fail-closed association or 0.1 mm physical-contact requirement is
changed. No MoveIt, robot controllers, physical motion or Workcell Studio
deployment occurs. Reaching 48 scenes or two steps does not qualify production.

## Production adapter follow-up

The explicit C++ RGB01/full-image adapter and reusable export verification are
specified in [the adapter contract](../../docs/segmentation_v2_adapter.md).
The smoke model remains rejected. Adapter evidence is separate from pilot model
quality and does not qualify a v2 checkpoint or alter the frozen dataset.

## Bounded full training campaign (001)

`campaign.py` supersedes the automatic CPU smoke fallback for full training.
It requires recorded preflight identities and measured full-run CPU feasibility,
then trains all 24 frozen training scenes for eight epochs. Its separate
`campaign.json` freezes the user improvement gate before measurement. It saves
model/optimizer/RNG checkpoints and raw predictions for the 12 validation scenes
only, selecting exclusively among eligible checkpoints. No automatic downloads.

Actual completed run and all failures: [campaign 001 evidence](../../evidence/segmentation_v2/campaign_001/README.md).
All eight checkpoints were rejected; export and C++ parity were therefore skipped.
This does not change the already verified adapter contract.

The exact executed host-specific preflight source and process evidence are retained
with the report. A new machine must measure feasibility against its own declared
budget and paths before launching; never reuse another host's timing admission.
The completed run used these commands (do not rerun the campaign into its existing
output directory):

```bash
python3 -m unittest discover -s tools/segmentation_v2 -p 'test_*.py'
python3 tools/segmentation_v2/campaign.py train --artifacts /home/user/epd-v2-artifacts/pilot-001 --weights /home/user/epd-v2-artifacts/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth --preflight /home/user/epd-v2-artifacts/campaign-001/preflight --output /home/user/epd-v2-artifacts/campaign-001/training
python3 tools/segmentation_v2/audit_campaign.py --artifacts /home/user/epd-v2-artifacts/pilot-001 --training /home/user/epd-v2-artifacts/campaign-001/training --output /home/user/epd-v2-artifacts/campaign-001/offline_audit.json
```

Offline audit performs no inference or checkpoint deserialization; it verifies
hashes and recomputes the frozen decision from genuine retained validation arrays.
It never enumerates held-out scene files.

## 300-scene training dataset expansion

[Executed expansion evidence](../../evidence/segmentation_v2/training_expansion_001/README.md)
records276 genuine new captures and final300/12/12 membership. Original pilot,
frozen evaluation memberships and rejected checkpoints remain intact.

`expansion.py` freezes an exclusive training-only plan and acquires six bounded
batches46 using the retained, hash-verified native executable. Attempts, progress
and batch seals are retained; the first failure stops capture. Existing attempts
cannot be silently retried. The read-only `verify` action checks immutable batch
and attempt hashes and reports remaining unattempted IDs; any failed-campaign
continuation requires separate explicit review/authorization.

For reproducible **new** acquisition, use a new exclusive external directory;
do not rerun successful captures in the completed directory:

```bash
python3 tools/segmentation_v2/expansion.py freeze --pilot /external/pilot-001 --emd-repo /read-only/EMD --output /external/new-expansion
python3 tools/segmentation_v2/expansion.py acquire --output /external/new-expansion
python3 tools/segmentation_v2/expansion.py verify --output /external/new-expansion
python3 tools/segmentation_v2/integrate_dataset.py integrate --pilot /external/pilot-001 --expansion /external/new-expansion --output /external/new-expansion/integration
python3 tools/segmentation_v2/integrate_dataset.py smoke --manifest /external/new-expansion/integration/dataset.json --output /external/new-expansion/integration/loader_smoke.json
python3 -m unittest discover -s tools/segmentation_v2 -p 'test_*.py'
```

The executed smoke was additionally wrapped by `pilot.bounded` at300 seconds/
4096 MiB; its exact command and process result are in the evidence. All metadata
bindings are validated before pixel access. `dataset.enumerate_records(manifest,
 'train')` enumerates300 records without opening evaluation pixels;
`train.verified_rows(manifest,'train')` yields the existing row/folder/report/IDs
interface, and `train.tensors` retains the original preprocessing and target
contract. Legacy pilot-directory calls remain unchanged. No training entrypoint
was run against the composed manifest; a future training campaign needs its own
budget and updated full-run assumptions.
