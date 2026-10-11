# EPD torchvision input/output contract

Status: **ADAPTER_READY_FOR_V2_EXPORT**. This qualifies the adapter only. The
CPU smoke checkpoint remains rejected and unexported; no trained v2 candidate
has passed an export comparison or deployment qualification.

## Explicit profiles and backwards compatibility

`session_config.json` accepts an optional, exact three-field `model_profile`:

```json
{
  "model_profile": {
    "preprocessing": "rgb01",
    "mask_layout": "full_image",
    "input_layout": "nchw"
  },
  "confidence_threshold": 0.8
}
```

Omitting `model_profile` keeps the legacy path. An explicit legacy declaration
uses `legacy_bgr_mean`, `roi`, and `legacy` respectively. Partial profiles,
unknown keys, unsupported combinations and ambiguous declarations throw an
exception. Precision level must be P3 for the RGB01 profile. A filename or output
name cannot enable it.

Legacy preprocessing still resizes the BGR8 image with OpenCV INTER_LINEAR,
converts to float32, subtracts BGR `(102.9801,115.9465,122.7717)` for float models,
pads zeros to the existing selected tensor size, and packs the existing CHW,
NCHW or NHWC layout. UINT8 models retain BGR values without mean subtraction.
Legacy masks remain ROI-relative probabilities, resized by the original
consumers. Float box scaling arithmetic is preserved, including rounding at
integer boundaries; detection truncates box edges, while localization/tracking
retain fractional edges and their existing floor/ceil geometry convention.
The legacy low-level confidence comparison remains `score > threshold`.

The opt-in profile accepts an OpenCV **BGR8** frame, reverses channels to RGB,
divides each byte by float32 255, and packs contiguous CHW storage with a batch
dimension of one. There is no mean subtraction, resize or padding. The frame and
model must have the **same fixed pixel raster**. EPDContainer bypasses legacy
minimum-side and padding calculations for this profile; any mismatch is rejected.
This intentionally does not implement implicit camera resampling, mask
resampling, or camera-intrinsics adjustment. Future training/export must use the
actual intended EPD camera raster, or an explicitly prepared, calibrated input.

All four runtime outputs are validated before decoding, including detection
counts, dimensions, finite values, score/probability range, classes, and box
extent. Invalid results throw; EPD's existing inference error handling prevents
publication of the invalid result. Valid empty tensors return zero detections.
Products use checked size_t arithmetic, and clipping precedes integer conversion.
Malformed legacy outputs now fail explicitly rather than using the unsafe 28x28
fallback or unchecked memory access; compatibility applies to valid legacy inputs.

RGB01 boxes are input-raster pixel coordinates, xyxy, clipped to image edges and
then truncated to integer half-open ROIs `[x0:x1,y0:y1]`, matching EPD detection
consumers. Localization and tracking use these same integer ROIs. Each full-image
mask is sliced exactly to `[y0:y1,x0:x1]` and cloned as float32 probabilities.
There is no threshold or interpolation during adaptation. Detection visualization,
point-cloud extraction, localization and tracking skip resizing for this profile
and reject ROI size mismatches. Existing consumers apply strict **mask > 0.5**.
RGB01 confidence filtering is inclusive **score >= threshold**. The detection
infer API retains its existing caller-filtered behavior (low-level threshold 0);
EPD's detection consumer applies the configured confidence threshold inclusively.
Localization/tracking receive the configured threshold directly. Defaults in
existing session configurations remain unchanged.

Geometry quality rules, conservative association and physical-contact
requirements are unchanged. Adapter readiness does not validate geometry,
association acceptance, or the 0.1 mm contact requirement.

## Required future ONNX interface

| Order | Name | Type | Shape | Meaning |
|---|---|---|---|---|
| input 0 | images | float32 | [1,3,H,W] | RGB01, fixed positive H/W, NCHW |
| output 0 | boxes | float32 | [N,4] | xyxy pixel coordinates on the input raster |
| output 1 | labels | int64 | [N] | foreground indices into EPD's label list |
| output 2 | scores | float32 | [N] | probabilities in [0,1] |
| output 3 | masks | float32 | [N,1,H,W] | full-image probabilities in [0,1] |

Use a background entry at index 0 in the EPD label file; cube foreground is 1
for this pilot. Every emitted label must be >=1 and < label count. Outputs use the
same ordering as torchvision, without class offsets. Background detections,
reversed/empty boxes, nonfinite tensors and invalid ranges are rejected.

A future export must support dynamic N, including `[0,4]`, `[0]`, `[0]`, and
`[0,1,H,W]`, with all output counts equal. The adapter also accepts fixed N,
including zero, for deterministic graph fixtures. Metadata may declare dynamic
mask channel/spatial dimensions (as the historical export does), but runtime
channel must be exactly one and spatial dimensions must equal the input raster.
Input batch/channel/spatial dimensions must be fixed as above. Supported names
alone are insufficient: metadata types/ranks/order and runtime dimensions are
independently checked. The installed C++ ONNX Runtime must load and execute the
whole graph; test fixtures use opset 11. An ONNX extension is no compatibility
claim.

The exported graph must include torchvision's own model transforms and restore
boxes/masks to this input raster. Do not add external ImageNet normalization to
RGB01, box fitting, erosion, oracle cropping, or a changed mask threshold.

For a future accepted checkpoint, retain its SHA256, weights licence, architecture,
exporter versions/opset and exact input preparation. In eval mode, save raw
PyTorch `boxes`, `labels`, `scores`, `masks` as an NPZ for each identical BGR input
prepared by the declared profile. Hash-pin the NPZ and ONNX. Run the reusable
verifier on positive and negative/zero-detection inputs:

```bash
python3 tools/segmentation_v2/verify_adapter.py candidate \
  --probe /external/adapter-build/epd_p3_contract_probe \
  --model /external/candidate.onnx --model-sha256 <sha256> \
  --image /external/exact-bgr-input.png \
  --reference /external/pytorch-raw.npz --reference-sha256 <sha256> \
  --classes 2 --atol 1e-5 --output /external/new-candidate-check
```

It compares **all raw outputs**, dtype, shape, labels and numerical differences
using Python ORT against the independently retained PyTorch outputs, then consumes
the model through the actual EPD C++ P3 inference path and compares confident
boxes/scores/labels plus exact ROI probabilities. Raw comparisons preserve
subpixel box values; EPD comparisons check documented integer ROIs. Numerical
comparison requires matching output order/detection association. It does not
silently align, fit or reorder predictions. This candidate route is prepared;
no trained v2 export exists to run it against yet.

## Build and reproduce checks

Use the existing ROS/vendor installation; no dependency or environment upgrades:

```bash
source /home/user/epd_ros2_ws/install/setup.bash
cmake -S easy_perception_deployment/easy_perception_deployment \
  -B /external/adapter-build -DEPD_DOWNLOAD_MODELS=OFF \
  -DEPD_REQUIRE_MODELS=OFF -DBUILD_TESTING=ON
cmake --build /external/adapter-build --target easy_perception_deployment \
  epd_test_p3_adapter epd_p3_contract_probe epd_test_geometry_quality \
  epd_test_observation epd_test_inference_scheduler -j2
ctest --test-dir /external/adapter-build \
  -R '^epd_test_(p3_adapter|geometry_quality|observation|inference_scheduler)$' \
  --output-on-failure
python3 tools/segmentation_v2/verify_adapter.py fixtures \
  --probe /external/adapter-build/epd_p3_contract_probe \
  --validation /external/pilot-001/validation-diagnostic-corrected \
  --output /external/new-adapter-fixture-check
python3 tools/segmentation_v2/verify_adapter.py historical \
  --probe /external/adapter-build/epd_p3_contract_probe \
  --model /external/cube_maskrcnn.onnx \
  --validation /external/pilot-001/validation-diagnostic-corrected \
  --dataset /external/pilot-001/dataset --output /external/new-historical-check
```

The fixture command hash-verifies the retained validation references and generates
clearly labelled constant-output **contract fixture graphs**, not trained models.
It tests 24 genuine retained prediction sets (12 baseline, 12 rejected-smoke),
58 confident instances, one zero/one detection dynamic graph, eight malformed
metadata/runtime graphs, legacy ROI semantics, and localization/tracking mask
consumption. It neither regenerates predictions from the smoke checkpoint nor
uses renderer IDs. Fixture RGB inputs are explicitly unit-test images. The
historical command performs one genuine inference on validation scene_025, using
its original RGB256 INTER_LINEAR512 RGB01 contract and hash-pinned model. It does
not use the separately retained historical two-cube regression or held-out split.

Every new probe process has a 60 s wall timeout, 45/50 s CPU soft/hard limits,
4 GiB address-space limit and two inference threads. Timeout kills the process
group and preserves stdout/stderr/process evidence. External output directories
must be new, protecting existing artifacts. No dataset/checkpoint is modified.

Execution evidence is under `evidence/segmentation_v2/adapter_001/`. The historical
check ran once before the final legacy rounding fix and additional probe-only
consumer checks; its exact executed binary is retained and hashed. The RGB01
path used for that historical inference is unchanged by the legacy rounding fix.
The final binary was separately verified against all contract fixtures.

Remaining action: supply an accepted trained v2 checkpoint and export on this
explicit interface, then run raw PyTorch/ONNX/C++ comparisons. There is no
remaining adapter contract blocker. Production qualification still requires the
previously documented strict segmentation and fail-closed association metrics.
No training, acquisition, held-out evaluation, motion, deployment, EMD edits,
PR merge or smoke-model export occurred in this adapter change.
