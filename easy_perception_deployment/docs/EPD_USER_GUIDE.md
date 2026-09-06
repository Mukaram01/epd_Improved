# Easy Perception Deployment (EPD) User Guide

## Overview

EPD turns camera images into ROS 2 perception results and provides a GUI for training, deployment, camera/model validation, live preview, 3D inspection, reproducible profiles/replay and performance-backend selection.

```text
TRAIN
Images → annotations → dataset validation → training → checkpoints → ONNX

DEPLOY
Camera → model → mode → health checks → live preview → ROS 2 results

3D / WORKCELL
RGB + aligned depth + CameraInfo → localization/tracking
→ normalized perceived objects → Workcell Studio / EMD
```

This fork is inspired by the original Easy Perception Deployment project:

https://easy-perception-deployment.readthedocs.io/en/latest/

The upstream documentation remains useful background, but this repository's GUI, code and local docs are the source of truth for current behaviour.

## Supported baseline

- Ubuntu 22.04
- ROS 2 Humble
- ONNX deployment models
- CPU as the portable/recovery backend
- Intel RealSense D435i-style RGB + aligned-depth + CameraInfo as the reference 3D camera path

CUDA, Jetson and TensorRT are optional EPD-8 paths and require target-machine validation.

---

# Quick Start — Deploy

1. Source ROS 2 Humble and the EPD workspace.
2. Start the camera node.
3. Open **Deploy**.
4. Select the ONNX model and matching labels.
5. Select or type the RGB camera topic.
6. Choose a perception mode.
7. Open **Camera Assistant** and verify the required streams.
8. Use **Smart Model Manager** to confirm model compatibility.
9. Review readiness.
10. Run perception and watch the **Live Perception View**.
11. For Localization/Tracking, inspect geometry with **3D Inspector**.
12. Save the working setup in **Profiles & Replay**.

Reference RealSense topics:

```text
RGB
/camera/camera/color/image_raw

Aligned depth
/camera/camera/aligned_depth_to_color/image_raw

CameraInfo
/camera/camera/color/camera_info
```

---

# EPD-0 — Camera Truth + Help

EPD distinguishes configuration from live discovery:

- **Detected** — selected RGB topic appeared in the latest ROS graph scan.
- **Configured** — a saved/manual topic exists but has not been verified live.
- **Missing** — no RGB input topic is configured.

A discovery timeout does not erase the configured camera topic.

The user-facing controls use clearer terms:

- **Detection overlay** = legacy visualization output;
- **Object masks** = segmentation-related output where supported.

Turning Detection overlay off does **not** stop ROS perception results.

Press **F1** from Launcher, Train or Deploy for offline help.

---

# EPD-1 — Camera Assistant

Open from Deploy or press:

```text
Ctrl+Shift+C
```

Camera Assistant checks:

- ROS 2 graph availability and distribution;
- image/CameraInfo topic discovery;
- selected RGB topic;
- inferred aligned-depth topic;
- inferred CameraInfo topic;
- whether each expected stream actually delivers a message;
- resolution and encoding where available;
- approximate topic rate where measurable;
- message age when timestamps are comparable to wall time.

States:

- **Live** — a message was sampled successfully;
- **No sample** — topic exists but no message arrived before timeout;
- **Missing** — expected topic not present.

For Classification, Counting and Color-Matching, RGB is the required stream.

For Localization and Tracking, RGB + aligned depth + CameraInfo are required.

The assistant never silently rewrites the selected camera topic.

---

# EPD-2 — Live Perception View

Deploy embeds the normal operator preview:

- while stopped: selected RGB camera stream;
- while running with Detection overlay enabled: EPD visualization output;
- while running with overlay disabled: camera stream remains visible while ROS perception results continue.

The view can expose:

- current image;
- object count;
- FPS where measurable;
- latency where published;
- frame age/staleness;
- truthful runtime states such as STOPPED, STARTING, LIVE, WAITING, STOPPING, FAILED or UNAVAILABLE.

A stale/missing preview must not remain labelled LIVE.

Normal operation should not require `rqt_image_view`, though ROS tools remain useful for debugging.

---

# EPD-3 — Smart Model Manager

Before Run, use model inspection to verify as much of the ONNX contract as the model actually exposes.

Checks include:

- ONNX validity;
- model inputs/outputs;
- task type where reliably inferable;
- label compatibility where verifiable;
- recommended/compatible perception modes;
- clear incompatibility errors.

Model metadata cannot always prove class ordering, training dataset or arbitrary post-processing semantics. Unverifiable properties must not be treated as confirmed.

## Choosing Faster R-CNN vs Mask R-CNN

**Faster R-CNN** is appropriate when bounding boxes are enough.

**Mask R-CNN** is preferable when instance masks/shape matter, especially for manipulation and irregular objects.

Typical deployment format:

```text
model.onnx
```

Typical training checkpoint formats:

```text
.pth / .pt
```

The selected label list must match the trained class order.

---

# EPD-4 — Training Studio

Training Studio improves the existing training workflow with:

- train/validation dataset statistics;
- annotation/class-count checks;
- warnings for common dataset structural problems;
- live iteration/loss/learning-rate/ETA parsing;
- validation AP only when the trainer actually emits it;
- checkpoint inventory;
- resume selected checkpoint;
- fresh-run path preserving archive-before-training behaviour;
- manual best-checkpoint selection;
- ONNX export and post-export model inspection.

Important distinctions:

- **Max iterations** decides when training ends.
- **Checkpoint interval** decides how often state is saved.
- Training loss alone does not prove generalization or overfitting.

See `EPD_TRAINING_STUDIO.md` for the detailed workflow.

---

# EPD-5 — Profiles & Replay

Open from Deploy or press:

```text
Ctrl+Shift+P
```

A profile captures a reproducible deployment configuration including:

- model path;
- model SHA256 when the asset exists;
- labels path;
- label SHA256 when the asset exists;
- RGB topic;
- use case;
- overlay/masks;
- confidence/max detections;
- backend/GPU index from EPD-8;
- other session settings.

Profile application is blocked while perception is running.

A **known-good** marker means the operator has accepted that exact configuration; the marker itself is not certification.

Deterministic replay uses the production inference/tracking path with repeatable sensor observations.

Reference replay:

```bash
ros2 launch easy_perception_deployment replay.launch.py \
  mode:=fast \
  summary_output:=/tmp/epd_replay_summary.json
```

Expected accepted result:

```json
"result": "PASS"
```

See `EPD_PROFILES_REPLAY.md`.

---

# EPD-6 — 3D Perception Inspector

Open from Deploy or press:

```text
Ctrl+Shift+3
```

The inspector is read-only. It shows evidence such as:

- result/depth frame dimensions;
- depth encoding;
- camera intrinsics;
- result/depth timestamp alignment;
- sampled valid-depth percentage;
- per-object centroid;
- observed dimensions;
- segmented point-cloud size;
- major axis/pose sanity;
- current stable Tracking IDs;
- lost Tracking IDs;
- production geometry counters from inference diagnostics.

The inspector does not replace production geometry truth with GUI heuristics.

Missing or invalid observed geometry must not be turned into guessed collision geometry.

Plane/background filtering is not enabled by default without measured evidence that the workcell requires it.

See `EPD_3D_PERCEPTION_TOOLS.md`.

---

# EPD-7 — Workcell Studio / EMD Contract

EPD can publish a normalized perception contract while keeping repository ownership separate.

Snapshot schema:

```text
workcell_perception_snapshot/v1
```

Status schema:

```text
workcell_perception_status/v1
```

Default topics:

```text
/workcell_studio/epd_detection_snapshot_json
/workcell_studio/epd_connector_status
```

The contract preserves:

- Workcell-supplied scene/camera identity;
- optional profile provenance;
- source timestamp/frame;
- exact stable Tracking IDs;
- lost IDs;
- observed pose/centroid;
- positive observed dimensions where available;
- health/status truth.

EPD-7 does not fabricate per-object confidence when the native 3D message cannot associate it reliably.

EPD remains perception-only. Workcell Studio / EMD retains scene, task, PlanningScene, grasp, MoveIt and robot-motion ownership.

See `EPD_WORKCELL_CONTRACT.md`.

---

# EPD-8 — Performance Backends

Open the Performance surface or press:

```text
Ctrl+Shift+B
```

Backends:

- **auto** — may use CUDA when the build/runtime supports it and fall back to CPU;
- **cpu** — portable reference/recovery path;
- **cuda** — explicit CUDA provider; failure is blocking rather than silently claiming GPU;
- **tensorrt** — explicit, gated path requiring a genuinely TensorRT-enabled vendor/runtime/image.

Build/provider capability probe:

```bash
ros2 run easy_perception_deployment epd_backend_probe
```

Benchmark:

```bash
ros2 run easy_perception_deployment epd_backend_benchmark.py \
  --backends cpu,cuda \
  --fixture <path-to-p8_tracking.json> \
  --output /tmp/epd_backend_benchmark.json
```

An accelerated backend is only acceptable after replay PASS and semantic comparison against the CPU baseline.

Jetson requires a compatible aarch64/JetPack/L4T or validated native path. Do not assume the x86 CUDA image is Jetson-compatible.

See `EPD_PERFORMANCE_BACKENDS.md`.

---

# EPD-9 — Release / Demo Quality

EPD-9 provides a repeatable handoff surface rather than relying on screenshots and terminal history alone.

## Release acceptance runner

Static checks:

```bash
ros2 run easy_perception_deployment epd_release_acceptance.py
```

Include ROS graph checks:

```bash
ros2 run easy_perception_deployment epd_release_acceptance.py --with-ros
```

Include deterministic replay:

```bash
ros2 run easy_perception_deployment epd_release_acceptance.py \
  --with-replay \
  --output /tmp/epd_release_acceptance.json
```

Statuses:

```text
PASS = required check passed
WARN = limitation/unavailable optional evidence needs review
FAIL = blocking release evidence failed
```

## Diagnostics bundle

```bash
ros2 run easy_perception_deployment epd_diagnostics_bundle.py \
  --output /tmp/epd_diagnostics.zip
```

Optionally include the selected EPD-5 profile:

```bash
ros2 run easy_perception_deployment epd_diagnostics_bundle.py \
  --profile ~/my_profile.epd-profile.json \
  --output /tmp/epd_diagnostics.zip
```

The bundle is read-only/best-effort and records missing/timed-out probes. HOME/user strings are redacted by default.

A diagnostics bundle and acceptance PASS are engineering evidence, not a functional-safety certificate.

For the complete handoff/demo sequence see:

- `EPD_RELEASE_DEMO_GUIDE.md`
- `EPD_ACCEPTANCE_CHECKLIST.md`
- `EPD_KNOWN_LIMITATIONS.md`
- `examples/profiles/README.md`

---

# Deploy Controls Reference

## Detection overlay

On = generate operator visualization output.

Off = reduce visualization overhead while ROS perception results continue.

## Object masks

Enable segmentation-related output where the model supports it. Useful when shape matters; can increase compute/message bandwidth.

## Confidence threshold

Minimum accepted detection score. `0.50` is a reasonable starting point, then tune using real workcell evidence.

## Max detections

Caps detections processed per frame to improve runtime predictability in crowded scenes.

## Image transport

- `raw` — simple, no compression work;
- `compressed` — lower network bandwidth with codec overhead.

---

# ROS 2 Inspection Commands

```bash
ros2 topic list -t
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/camera/color/camera_info --once
ros2 topic hz /easy_perception_deployment/image_output
ros2 topic echo /easy_perception_deployment/epd_tracking_output --once
ros2 topic echo /easy_perception_deployment/inference_diagnostics --once
```

For Workcell contract output:

```bash
ros2 topic echo /workcell_studio/epd_detection_snapshot_json --once
ros2 topic echo /workcell_studio/epd_connector_status --once
```

---

# Troubleshooting

## Camera list empty

1. Source ROS 2 and the workspace.
2. Confirm the camera node is running.
3. Run `ros2 topic list -t`.
4. Click Refresh topics.
5. Open Camera Assistant.
6. Type the expected RGB topic manually if discovery is unavailable.

## Configured but not Detected

The topic is saved but was not verified in the latest graph scan. Start the camera and refresh.

## Detected but No sample

The topic exists on the graph but did not deliver before the assistant timeout. Check publisher state, QoS and `ros2 topic hz`.

## No detections

Check:

- live image;
- correct ONNX model;
- matching label list;
- object class exists in the model;
- confidence is not too high;
- model-manager compatibility state.

## Localization/Tracking problems

Also check:

- RGB live;
- aligned depth live;
- CameraInfo live;
- depth/color alignment;
- valid depth range;
- 3D Inspector geometry counters.

## Explicit CUDA/TensorRT fails

This is preferable to silently claiming acceleration. Open Performance Backends, run `epd_backend_probe`, and review `EPD_PERFORMANCE_BACKENDS.md`.

---

# Recommended Release Workflow

```text
1. Build/source Humble workspace
2. Start camera
3. Camera Assistant PASS
4. Model Manager PASS
5. Run Tracking
6. Live preview verified
7. 3D Inspector verified
8. Save target-machine profile
9. Deterministic replay PASS
10. Workcell contract checked (when used)
11. Backend benchmark checked (when accelerated)
12. epd_release_acceptance.py --with-replay
13. epd_diagnostics_bundle.py
14. Capture reviewed screenshots
15. Archive evidence + known limitations
```

No real robot motion is required to accept EPD itself.

---

# Product Roadmap

See:

```text
docs/EPD_PRODUCT_ROADMAP.md
```

EPD-0 through EPD-9 now form the complete productization sequence:

```text
camera truth
→ camera health
→ live feedback
→ model truth
→ training observability
→ profiles/replay
→ 3D diagnostics
→ Workcell contract
→ performance backends
→ release/demo evidence
```
