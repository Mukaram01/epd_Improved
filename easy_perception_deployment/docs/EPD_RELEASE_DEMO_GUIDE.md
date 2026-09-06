# EPD Release & Demo Guide

This is the handoff/demo entry point for the EPD-0 → EPD-9 productization line.

EPD is the perception subsystem. It does not own Workcell Studio scenes, task intent, PlanningScene updates, grasp selection, MoveIt execution, or robot motion.

## Supported baseline

- Ubuntu 22.04
- ROS 2 Humble
- CPU as the portable/recovery inference path
- Intel RealSense D435i defaults for the reference 3D camera workflow
- ONNX deployment models

CUDA, Jetson and TensorRT are optional EPD-8 acceleration paths and must be validated on the target machine before use.

## Demo goal

A successful standard demo should prove this operator flow:

```text
Launcher
  ↓
Deploy
  ↓
Camera Assistant verifies RGB/depth/CameraInfo
  ↓
Model Manager validates model + labels
  ↓
Tracking mode
  ↓
Live Perception View shows current camera/perception state
  ↓
3D Inspector shows geometry + stable IDs
  ↓
Profiles & Replay captures/replays the configuration
  ↓
Workcell contract publishes normalized perceived objects
```

The demo does not need robot motion to prove EPD acceptance.

## 1. Build and source

From the ROS 2 workspace root:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

For a fresh workstation, install dependencies before the build and ensure the required ONNX model assets are present. See `README.md` and `EPD_USER_GUIDE.md`.

## 2. Start the reference camera

For the RealSense D435i reference path, the expected streams are:

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/camera_info
```

Verify the graph:

```bash
ros2 topic list -t
```

## 3. Launch the GUI

From the package GUI directory use the normal launcher for this repository.

Expected first operator checks:

- Launcher opens without an exception.
- Help & Guides is reachable with F1.
- Deploy opens.
- Camera Input preserves the configured topic even if discovery temporarily times out.
- Camera Assistant can distinguish configured, detected and live streams.

## 4. Validate the camera

Open **Camera Assistant** (`Ctrl+Shift+C`).

For Tracking/Localization, confirm:

- RGB: live
- aligned depth: live
- CameraInfo: live
- resolution/encoding are sensible
- message age/rate are current when measurable

A stopped camera must not be shown as live.

## 5. Validate the model

Open the EPD-3 model-management surface and confirm:

- ONNX file exists and can be inspected;
- selected labels match the model contract where the model exposes enough metadata to verify this;
- selected perception mode is compatible;
- no unsupported assumption is presented as verified truth.

For the reference COCO 3D demo, Mask R-CNN + COCO labels is a suitable starting point when the model assets are available.

## 6. Run Tracking

Recommended reference settings:

```text
Mode                  Tracking
Detection overlay     On for operator validation
Object masks          On when model supports masks
Backend               CPU for baseline acceptance
Confidence            0.50 initially
RGB topic             /camera/camera/color/image_raw
```

Start perception.

The normal operator loop should remain inside Deploy; `rqt_image_view` is not required for the standard demo.

## 7. Inspect live perception

The EPD-2 Live Perception View should show current RGB/perception imagery and truthful runtime state.

Verify:

- preview changes with the live camera;
- object count is derived from actual EPD outputs;
- FPS/latency display degrades to `—` when evidence is unavailable;
- stale input never remains labelled LIVE;
- turning Detection overlay off does not stop ROS perception results.

## 8. Inspect 3D truth

Open **3D Inspector** (`Ctrl+Shift+3`).

Verify:

- result/depth dimensions agree;
- depth encoding is supported;
- intrinsics are finite/positive;
- result timestamp and embedded depth timestamp align;
- valid-depth sampling is plausible;
- localized objects expose centroid/dimensions/point-cloud evidence when available;
- Tracking IDs remain stable across updates;
- lost IDs are surfaced exactly as published by EPD.

Do not infer collision geometry when EPD did not observe valid dimensions.

## 9. Save a reproducible profile

Open **Profiles & Replay** (`Ctrl+Shift+P`).

Save the current configuration and verify that the profile captures:

- model path + SHA256 when available;
- labels path + SHA256 when available;
- camera topic;
- use case;
- overlay/masks;
- confidence/max detections;
- EPD-8 backend + GPU index.

Mark a profile known-good only after the target workstation/camera/model combination has actually passed acceptance.

## 10. Run deterministic replay

The deterministic replay path is the preferred release regression because it exercises production inference/tracking while keeping sensor input repeatable.

```bash
ros2 launch easy_perception_deployment replay.launch.py \
  mode:=fast \
  summary_output:=/tmp/epd_replay_summary.json
```

Expected result:

```json
"result": "PASS"
```

The summary includes stable-ID/lifecycle, geometry, backpressure and EPD-8 performance evidence.

## 11. Validate the Workcell contract

For an already-running EPD Tracking pipeline:

```bash
ros2 launch easy_perception_deployment workcell_contract.launch.py \
  scene_id:=demo_scene \
  camera_id:=realsense_d435i_1 \
  source_mode:=tracking \
  require_tracking_ids:=true
```

Inspect:

```bash
ros2 topic echo /workcell_studio/epd_detection_snapshot_json --once
ros2 topic echo /workcell_studio/epd_connector_status --once
```

Expected contract:

```text
workcell_perception_snapshot/v1
workcell_perception_status/v1
```

The contract must preserve EPD timestamp/frame/stable-ID truth and must not invent confidence or geometry.

## 12. Optional backend benchmark

CPU remains the reference baseline. On an accelerated machine:

```bash
ros2 run easy_perception_deployment epd_backend_probe

ros2 run easy_perception_deployment epd_backend_benchmark.py \
  --backends cpu,cuda \
  --fixture <path-to-p8_tracking.json> \
  --output /tmp/epd_backend_benchmark.json
```

Adopt an accelerated backend only when:

1. replay PASSes;
2. semantic comparison with CPU is acceptable;
3. performance improvement is measured on the target hardware.

## 13. Generate acceptance evidence

Static release checks:

```bash
ros2 run easy_perception_deployment epd_release_acceptance.py
```

With ROS graph checks:

```bash
ros2 run easy_perception_deployment epd_release_acceptance.py --with-ros
```

With deterministic replay:

```bash
ros2 run easy_perception_deployment epd_release_acceptance.py \
  --with-replay \
  --output /tmp/epd_release_acceptance.json
```

A `WARN` report can still be useful evidence when optional hardware/services are absent. A `FAIL` must be resolved before calling the tested configuration release-ready.

## 14. Generate a diagnostics bundle

```bash
ros2 run easy_perception_deployment epd_diagnostics_bundle.py \
  --output /tmp/epd_diagnostics.zip
```

Optionally attach a profile:

```bash
ros2 run easy_perception_deployment epd_diagnostics_bundle.py \
  --profile ~/my_profile.epd-profile.json \
  --output /tmp/epd_diagnostics.zip
```

The collector is read-only and best-effort. By default it redacts HOME/user strings from captured text. Use `--include-paths` only when full local paths are intentionally needed.

The bundle includes available evidence such as:

- EPD config files;
- recent EPD GUI logs;
- ROS nodes/topics;
- inference diagnostics sample;
- Tracking sample;
- Workcell connector status sample;
- backend capability probe;
- Docker/NVIDIA environment evidence;
- manifest describing unavailable/timed-out checks.

A diagnostics bundle is evidence, not a safety certificate.

## 15. Screenshots to capture for a release/demo

Capture these manually on the target machine:

1. Launcher with Train / Deploy / Help.
2. Deploy with truthful Camera Input state.
3. Camera Assistant healthy RGB/depth/CameraInfo.
4. Smart Model Manager with validated model.
5. Live Perception View with a visible detection.
6. 3D Inspector showing geometry/stable tracking ID.
7. Profiles & Replay with the known-good profile selected.
8. Performance Backends showing the selected backend and environment evidence.
9. Workcell contract snapshot/status in a terminal or downstream viewer.
10. Release acceptance report showing the final status.

Do not commit screenshots containing sensitive paths, usernames, customer data, or proprietary workcell imagery without review.

## Release evidence set

For a reproducible handoff, archive together:

```text
EPD commit SHA
epd_release_acceptance.json
epd_diagnostics.zip
known-good .epd-profile.json
replay summary JSON
backend benchmark JSON (when acceleration is used)
screenshots from the target workstation
notes for any accepted WARN items
```

Also review `EPD_ACCEPTANCE_CHECKLIST.md` and `EPD_KNOWN_LIMITATIONS.md` before declaring a demo/release configuration ready.
