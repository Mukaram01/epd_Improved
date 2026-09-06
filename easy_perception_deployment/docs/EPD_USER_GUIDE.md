# Easy Perception Deployment (EPD) User Guide

## Overview

EPD turns camera images into ROS 2 perception results and provides a GUI for training and deployment.

```text
Camera / Images
      ↓
Dataset preparation
      ↓
Model training
      ↓
ONNX export
      ↓
Deployment
      ↓
ROS 2 perception output
```

This fork is inspired by the original Easy Perception Deployment project. The original documentation remains useful reference material:

https://easy-perception-deployment.readthedocs.io/en/latest/

The current fork has evolved beyond the upstream documentation, so the local GUI, repository documentation and runtime configuration are the source of truth for current behaviour.

---

# Quick Start — Deploy

1. Start ROS 2 and the camera node.
2. Open **Deploy**.
3. Select the ONNX model.
4. Select the matching label list.
5. Select or type the RGB camera topic.
6. Choose a perception mode.
7. Open **Camera Assistant** and verify the camera health needed by the selected mode.
8. Review the readiness state.
9. Run perception.

For a RealSense D435i the common topics in this fork are:

```text
RGB:
/camera/camera/color/image_raw

Aligned depth:
/camera/camera/aligned_depth_to_color/image_raw

CameraInfo:
/camera/camera/color/camera_info
```

## Camera state: Detected vs Configured

EPD-0 distinguishes configuration from live discovery:

- **Detected** — the selected RGB image topic appeared in the latest ROS 2 image-topic scan.
- **Configured** — a saved/manual topic exists, but the latest scan did not verify that it is currently live.
- **Missing** — no RGB input topic is configured.

A saved topic is preserved if ROS topic discovery times out or the camera has not started yet. You can also type the expected topic manually.

Useful checks:

```bash
ros2 topic list -t
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/camera/color/camera_info --once
```

---

# Camera Assistant — EPD-1

The Camera Assistant is the dedicated camera-health view. Open it from the **Camera Input** card in Deploy or press:

```text
Ctrl+Shift+C
```

The assistant checks:

- whether the ROS 2 CLI and graph can be queried;
- the current ROS distribution;
- detected `sensor_msgs/msg/Image` topics;
- detected `sensor_msgs/msg/CameraInfo` topics;
- the selected RGB topic;
- an aligned-depth topic inferred from the camera namespace or RealSense defaults;
- a CameraInfo topic inferred from the camera namespace or RealSense defaults;
- whether each required topic actually delivers a sample;
- resolution and encoding when available;
- approximate topic rate when available;
- message-header age when ROS timestamps are comparable to wall time.

The assistant does **not** silently rewrite your Deploy camera configuration.

## Stream states

- **Live** — topic exists and a message was sampled successfully.
- **No sample** — topic exists on the ROS graph but no message arrived before the health-check timeout.
- **Missing** — the expected topic is not present on the ROS graph.

## 2D versus 3D requirements

For these modes, RGB is required while depth and CameraInfo are treated as optional diagnostics:

- Classification;
- Counting;
- Color-Matching.

For these modes, all three camera inputs are treated as required:

- Localization;
- Tracking.

For 3D operation, use aligned depth whenever possible. The normal RealSense topic is:

```text
/camera/camera/aligned_depth_to_color/image_raw
```

CameraInfo provides the camera intrinsics used by geometry calculations.

## What the assistant should look like

Typical healthy RealSense result:

```text
ROS 2             Connected (humble)
RGB               640×480 @ ~30 Hz
Depth             640×480 @ ~30 Hz / aligned
CameraInfo        live
Selected RGB      /camera/camera/color/image_raw
Last frame age    low / current
```

If a stream is missing, the **What to do next** card gives mode-specific remediation rather than only reporting a failure.

---

# Deploy Controls

## Detection overlay

This is the user-facing name for the legacy visualization mode.

**On**
- generate visualization output for human inspection.

**Off**
- reduce visualization overhead.
- ROS perception results still publish.

Turning the detection overlay off does **not** mean perception is disabled.

## Object masks

Object masks control segmentation-related per-object output where supported.

Useful for:
- Mask R-CNN;
- irregular objects;
- manipulation where precise object shape matters.

Mask/segmentation output can increase compute and message bandwidth.

## Confidence threshold

Minimum confidence accepted as a detection.

Start around:

```text
0.50
```

Then tune against your real camera/workcell:
- raise it to reduce false positives;
- lower it if difficult valid objects are missed.

## Max detections

Caps how many detections are processed per frame. A sensible cap can improve runtime predictability in crowded scenes.

## CPU vs GPU

Start with **CPU** unless the deployment environment and inference backend are already configured for GPU acceleration.

## Image transport

- `raw` — simple, avoids compression overhead;
- `compressed` — lower network bandwidth but adds encode/decode work.

---

# Perception Modes

## Classification

Run the base model inference path.

## Counting

Count/filter selected classes.

## Color-Matching

Compare detected objects against a reference colour template.

## Localization

Add 3D geometry/position information using depth.

## Tracking

Localization plus persistent object IDs across frames.

For Workcell Studio manipulation workflows, **Localization** and **Tracking** are usually the most relevant because downstream grasp planning needs 3D scene information.

---

# Training a Model

## 1. Prepare your dataset

A dataset contains images and annotations:

```text
dataset/
├── images/
└── annotations/
```

Training data should represent the real workcell:
- expected lighting;
- backgrounds;
- object orientations;
- partial occlusions;
- normal camera distance and viewpoint.

The labels used during training must match the label list selected in EPD.

---

# Choosing an Architecture

## Faster R-CNN

Use Faster R-CNN when bounding-box object detection is sufficient.

Good for:
- general object detection;
- applications where the object box is enough;
- avoiding segmentation complexity when it is not needed.

Typical output concept:

```text
class
bounding box
confidence
```

## Mask R-CNN

Use Mask R-CNN when instance segmentation is required.

Good for:
- robot grasping;
- irregular objects;
- precise object boundaries;
- workflows that use masks/segmentation.

Typical output concept:

```text
class
bounding box
pixel mask
confidence
```

---

# Training Parameters

## Max Iterations

The total number of learning/optimizer steps.

Too low:
- model may not learn enough.

Too high:
- longer training;
- possible overfitting;
- unnecessary compute.

## Checkpoint Interval

How often the training state is saved.

Example:

```text
Checkpoint interval = 500

500
1000
1500
2000
```

**Max iterations and checkpoint interval are different controls.**

- max iterations decides when training ends;
- checkpoint interval decides how often recovery points are written.

## Learning Rate Steps

Controls when learning becomes more gradual.

## Test Period

Controls how often the model is evaluated during training.

---

# Deployment Model

EPD deployment uses ONNX models.

Recommended deployment format:

```text
model.onnx
```

Why ONNX?
- portable;
- supported by inference engines;
- independent from the original training framework.

## Labels

The label list must match the training classes and ordering.

Example:

```text
cup
box
tool
```

Incorrect ordering can produce incorrect displayed class names.

---

# Model Formats

| Extension | Purpose |
|---|---|
| `.onnx` | Recommended EPD deployment/interchange format |
| `.pth` | Common PyTorch training checkpoint |
| `.pt` | Common PyTorch model format |
| `.engine` | TensorRT-specific optimized engine |

Typical workflow:

```text
Training: .pth / .pt
Deployment: .onnx
```

---

# Where to Get Pretrained Models

Common sources include:
- ONNX model repositories;
- PyTorch / torchvision model collections;
- NVIDIA model resources.

Always verify:
- architecture compatibility;
- input size and preprocessing;
- output contract;
- class labels/order;
- ONNX compatibility.

---

# Troubleshooting

## Camera list is empty

1. Confirm ROS 2 and the workspace are sourced.
2. Confirm the camera node is running.
3. Run `ros2 topic list -t`.
4. Click **Refresh topics**.
5. Open **Camera Assistant** for a full graph/sample health check.
6. If discovery is unavailable, type the expected RGB topic manually.

EPD preserves a saved topic instead of clearing it when discovery fails.

## Topic is Configured but not Detected

The topic is saved but was not verified in the latest ROS graph scan. This can be normal if the camera starts later. Start the camera and refresh again.

## Topic is Detected but Camera Assistant says No sample

The topic name exists on the ROS graph, but the health probe did not receive a message before timeout. Check:

- whether the camera publisher is actually streaming;
- camera driver state;
- publisher/subscriber QoS compatibility;
- `ros2 topic hz <topic>`;
- whether the topic name is stale from another node.

## No detections

Check:
- camera image is live;
- correct model selected;
- correct matching labels selected;
- object exists in the model classes;
- confidence threshold is not too high.

## Tracking/localization problems

Also check:
- Camera Assistant reports RGB live;
- aligned depth is live;
- CameraInfo is live;
- depth is aligned with colour;
- object is inside valid depth range.

---

# ROS 2 Outputs

Useful inspection commands:

```bash
ros2 topic list
ros2 topic hz /easy_perception_deployment/image_output
ros2 topic echo /easy_perception_deployment/epd_tracking_output --once
```

The exact mode-specific output depends on the selected perception mode.

---

# EPD + Workcell Studio / EMD

EPD remains the perception subsystem. Workcell Studio owns scene/workcell definition, task definition, planning and simulation.

```text
Camera
  ↓
EPD perception
  ↓
normalized perceived objects
  ↓
Workcell Studio / EMD
  ↓
grasp planning
  ↓
MoveIt
```

Tracking is especially useful because stable object IDs help downstream systems reason about the same object across frames.

---

# Recommended Beginner Workflow

1. Start with a trusted pretrained model.
2. Confirm the camera stream first with **Camera Assistant**.
3. Test deployment with the correct labels.
4. Use detection overlay while validating the pipeline.
5. Prepare a small representative labelled dataset.
6. Validate the dataset.
7. Train with moderate iterations and regular checkpoints.
8. Export/select the ONNX model.
9. Deploy with the same labels and camera topic.
10. Move to Localization/Tracking when 3D manipulation needs it.
11. Confirm RGB, aligned depth and CameraInfo before 3D operation.
12. Disable optional visualization/mask overhead only after the workflow is understood and verified.

---

# Product Roadmap

See:

```text
docs/EPD_PRODUCT_ROADMAP.md
```

EPD-0 and EPD-1 establish truthful camera configuration and camera-health diagnostics. The next phase is EPD-2: an embedded live perception preview so normal operation no longer requires `rqt_image_view`.
