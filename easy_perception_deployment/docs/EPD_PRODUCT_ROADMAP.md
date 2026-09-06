# EPD Product Roadmap

This roadmap records the EPD-0 → EPD-9 productization sequence for the current Easy Perception Deployment fork.

EPD remains the perception subsystem. Workcell Studio / Easy Manipulator retains scene, task, PlanningScene, grasp, MoveIt and robot-motion ownership.

## Product flow

```text
TRAIN
Images → labels → dataset validation → training → checkpoints → ONNX export

DEPLOY
Camera → model → perception mode → health/preview → run → ROS 2 outputs

3D / WORKCELL
RGB + depth + CameraInfo → localization/tracking
→ normalized perceived objects → Workcell Studio / EMD → grasp planning → MoveIt
```

## EPD-0 — Camera truth + Help v2

Status: **implemented** by `feature/epd0-camera-truth-help-v2`.

Delivered:
- preserve configured RGB topic when ROS discovery fails/times out;
- distinguish Configured / Detected / Missing truth;
- clearer Detection overlay and Object masks terminology;
- contextual help for camera, mode, transport, CPU/GPU, confidence and limits;
- Help & Guides from launcher/Deploy/F1;
- expanded RealSense/training/deployment/troubleshooting guidance.

Acceptance truth: a configured topic is not presented as live merely because it is saved.

## EPD-1 — Camera Assistant

Status: **implemented** by `feature/epd1-camera-assistant`.

Delivered:
- ROS graph/distribution health;
- RGB/depth/CameraInfo discovery and live sampling;
- resolution, encoding, rate and message-age evidence where available;
- mode-aware 2D vs 3D stream requirements;
- compact camera-health summary in Deploy;
- `Ctrl+Shift+C` access.

Acceptance truth: stopped/unresponsive streams must not remain labelled live.

## EPD-2 — Live Perception View

Status: **implemented** by `feature/epd2-live-perception-view`.

Delivered:
- embedded camera/perception preview in Deploy;
- EPD image output when overlay is enabled;
- camera fallback while overlay is disabled;
- object count, FPS, latency and frame-age evidence where available;
- truthful STOPPED/STARTING/LIVE/WAITING/STOPPING/FAILED/UNAVAILABLE states;
- normal operation no longer depends on `rqt_image_view`.

Acceptance truth: missing/stale preview never remains labelled LIVE.

## EPD-3 — Smart Model Manager

Status: **implemented** by `feature/epd3-smart-model-manager`.

Delivered:
- ONNX validity and model I/O inspection;
- task/model capability classification where reliably inferable;
- label compatibility checks where evidence exists;
- pretrained/bundled model guidance;
- mode recommendations and incompatibility blocking.

Acceptance truth: unverifiable model metadata is not presented as certainty.

## EPD-4 — Training Studio

Status: **implemented** by `feature/epd4-training-studio`.

Delivered:
- train/validation dataset statistics;
- annotation/class structural warnings;
- live training progress parsing;
- checkpoint inventory and resume/fresh-run controls;
- manual export-checkpoint selection;
- ONNX export + EPD-3 validation;
- beginner guidance around loss vs actual validation evidence.

See `docs/EPD_TRAINING_STUDIO.md`.

Acceptance truth: training loss alone is not used to claim model quality or overfitting.

## EPD-5 — Profiles + Replay

Status: **implemented** by `feature/epd5-profiles-replay`.

Delivered:
- named/versioned perception profiles;
- model/label SHA256 provenance when assets exist;
- safe portable asset relocation by basename + hash;
- import/export and known-good marker;
- profile application blocked while perception runs;
- deterministic fixture replay and rosbag inspection/playback;
- reproducible configuration including EPD-8 backend fields.

See `docs/EPD_PROFILES_REPLAY.md`.

Acceptance truth: a known-good flag is operator provenance, not certification.

## EPD-6 — 3D Perception Tools

Status: **implemented** by `feature/epd6-3d-perception-tools`.

Delivered:
- read-only 3D Inspector (`Ctrl+Shift+3`);
- depth/result dimension, encoding, intrinsics and timestamp checks;
- sampled valid-depth ratio;
- centroid, dimensions, point-cloud, axis and pose sanity evidence;
- stable/lost Tracking IDs;
- production geometry counters;
- stale-output detection.

See `docs/EPD_3D_PERCEPTION_TOOLS.md`.

Acceptance truth: plane/background filtering remains off until a measured workcell failure justifies changing perception semantics.

## EPD-7 — Workcell Studio / EMD Contract

Status: **implemented** by `feature/epd7-workcell-contract`.

Delivered:
- `workcell_perception_snapshot/v1` normalized snapshot;
- `workcell_perception_status/v1` health/status;
- exact Tracking IDs/lost IDs preserved;
- exact source timestamp/frame provenance;
- observed pose/centroid/dimensions without invented geometry;
- Workcell-supplied scene/camera/profile provenance;
- standalone/live/replay contract paths;
- JSON schemas and validation.

See `docs/EPD_WORKCELL_CONTRACT.md`.

Acceptance truth: EPD-7 contains no scene authoring, PlanningScene write, grasp selection, MoveIt call or robot-motion command.

## EPD-8 — Performance Backends

Status: **implemented** by `feature/epd8-performance-backends`.

Delivered:
- explicit `auto`, `cpu`, `cuda`, `tensorrt` backend selection + GPU index;
- legacy `useCPU` migration compatibility;
- actual ONNX Runtime provider selection;
- explicit CUDA failure instead of silent CPU claims;
- conditionally compiled/gated TensorRT integration;
- host/Docker/NVIDIA/build capability evidence;
- `epd_backend_probe`;
- guarded CUDA/Jetson build helper;
- deterministic CPU vs accelerated benchmark;
- stable-ID/lifecycle/geometry semantic comparison against CPU baseline.

See `docs/EPD_PERFORMANCE_BACKENDS.md`.

Acceptance truth: TensorRT remains unavailable until the vendor/runtime/image genuinely provide TensorRT support. CPU remains the portable recovery path.

## EPD-9 — Release / Demo Quality

Status: **implemented** by `feature/epd9-release-demo-quality`.

Delivered:
- `epd_release_acceptance.py` for machine-readable PASS/WARN/FAIL handoff checks;
- optional ROS graph/backend probe and deterministic replay in acceptance;
- `epd_diagnostics_bundle.py` for read-only best-effort diagnostics ZIP generation;
- default HOME/user redaction in diagnostics capture;
- refreshed EPD user guide covering EPD-0 through EPD-9;
- reproducible release/demo guide;
- explicit release acceptance checklist;
- known-limitations/evidence-boundary document;
- portable reference RealSense Tracking CPU profile template;
- installed docs/examples alongside package runtime assets;
- in-app Release & Demo / Diagnostics / Acceptance help topics;
- `Ctrl+Shift+R` shortcut to the release help entry;
- regression tests for release helpers and reference profile shape.

Primary release artifacts:

```text
docs/EPD_RELEASE_DEMO_GUIDE.md
docs/EPD_ACCEPTANCE_CHECKLIST.md
docs/EPD_KNOWN_LIMITATIONS.md
docs/EPD_USER_GUIDE.md
examples/profiles/realsense_tracking_cpu.epd-profile.json
```

Recommended evidence commands:

```bash
ros2 run easy_perception_deployment epd_release_acceptance.py \
  --with-replay \
  --output /tmp/epd_release_acceptance.json

ros2 run easy_perception_deployment epd_diagnostics_bundle.py \
  --output /tmp/epd_diagnostics.zip
```

Acceptance truth:
1. Release acceptance can run static-only, with ROS checks, or with deterministic replay.
2. Missing optional hardware/tools become WARN/evidence rather than fabricated PASS.
3. Blocking config/model/replay failures remain FAIL.
4. Diagnostics collection is read-only and records unavailable/timed-out probes.
5. The release bundle does not claim functional-safety certification.
6. Reference profiles are templates until the exact target configuration is accepted and captured with real asset hashes.
7. Screenshots must be reviewed for sensitive paths/customer imagery before external sharing.
8. EPD release acceptance requires no robot motion and does not weaken downstream safety gates.

## Roadmap closure / next-development rule

The planned productization loop is now complete:

```text
camera truth
→ camera health
→ live feedback
→ model truth
→ training observability
→ profiles/replay
→ 3D diagnostics
→ Workcell Studio contract
→ performance backends
→ release/demo evidence
```

Future increments should be driven by measured acceptance failures, target-customer requirements or validated performance bottlenecks rather than adding features without evidence.
