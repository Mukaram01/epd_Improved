# EPD Product Roadmap

This roadmap turns the current Easy Perception Deployment fork into a clearer industrial perception product while keeping EPD separate from Workcell Studio scene/task ownership.

## Product flow

```text
TRAIN
Images → labels → dataset validation → training → checkpoints → ONNX export

DEPLOY
Camera → model → perception mode → preview/validate → run → ROS 2 outputs

3D / WORKCELL
RGB + depth + CameraInfo → localization/tracking → normalized perceived objects → Workcell Studio / EMD → grasp planning → MoveIt
```

## EPD-0 — Camera truth + Help v2

Status: implemented by the `feature/epd0-camera-truth-help-v2` increment.

Goals:
- preserve the saved/manual RGB topic when ROS topic discovery fails or times out;
- distinguish a topic that is **Configured** from one actually **Detected** on the ROS 2 graph;
- remove the contradictory `Not configured` + `Ready` presentation;
- make camera discovery messages operator-friendly;
- use a more tolerant ROS image-topic scan timeout;
- rename `Visual output` to `Detection overlay` without changing backend semantics;
- rename `Segmentation` to `Object masks` in the refreshed Deploy UI;
- add practical tooltips for camera, mode, overlay, masks, image transport, CPU/GPU, confidence and limits;
- expose Help & Guides from the launcher, Deploy header and F1;
- expand in-app guidance for RealSense, training, deployment, perception modes, troubleshooting and Workcell Studio integration;
- link the original upstream EPD ReadTheDocs documentation as reference material.

Acceptance:
1. A saved camera topic remains visible after discovery timeout.
2. The readiness chip says `Detected`, `Configured`, or `Missing` truthfully.
3. An unverified configured topic does not make the header falsely imply live-camera readiness.
4. Detection overlay help makes clear that turning it off does not disable ROS perception results.
5. F1 opens the in-app guide from Launcher, Train and Deploy.

## EPD-1 — Camera Assistant

Status: implemented by the `feature/epd1-camera-assistant` increment.

Adds a dedicated camera-health surface:
- ROS 2 environment/graph status and ROS distribution;
- detected `sensor_msgs/msg/Image` and `sensor_msgs/msg/CameraInfo` topics;
- selected RGB stream health;
- inferred RealSense/custom aligned-depth stream health;
- inferred CameraInfo stream health;
- live sample verification rather than graph presence alone;
- resolution and encoding where available;
- measured topic rate where available;
- latest message header age where the ROS clock is comparable to wall time;
- explicit 3D requirements for Localization and Tracking;
- actionable remediation when RGB, aligned depth or CameraInfo is missing;
- a compact camera-health summary embedded back into Deploy;
- one-click `Camera Assistant` access from the Camera Input card;
- `Ctrl+Shift+C` shortcut from Deploy.

Target operator view:

```text
ROS 2             Connected (humble)
RGB               640×480 @ 30 Hz
Depth             aligned / live
CameraInfo        available
Selected RGB      /camera/camera/color/image_raw
Last frame age    < 100 ms
```

Acceptance:
1. Camera Assistant opens without blocking the Deploy UI.
2. With no ROS 2 CLI available, the assistant reports ROS unavailable and gives remediation.
3. With a saved RGB topic but stopped camera, RGB is shown as missing/unresponsive rather than live.
4. With RealSense publishing, RGB, aligned depth and CameraInfo are identified and sampled.
5. Resolution, encoding, rate and message age are shown when the underlying ROS tools provide them.
6. Localization/Tracking treats depth and CameraInfo as required; 2D modes label them optional.
7. A successful assistant scan updates the existing EPD-0 camera truth/cache without changing the selected topic.
8. The assistant never silently rewrites camera configuration.

## EPD-2 — Live Perception View

Embed a preview inside Deploy:
- RGB image;
- optional detection boxes/masks;
- object count;
- FPS and latency;
- clear start/stop state;
- no need to open `rqt_image_view` for the normal operator workflow.

## EPD-3 — Smart Model Manager

Before Run, inspect and explain models:
- ONNX validity;
- model/task type where it can be determined reliably;
- input/output compatibility;
- label count/order checks where possible;
- bundled/pretrained model library;
- recommended perception modes;
- clear incompatibility errors.

## EPD-4 — Training Studio

Make Train observable and recoverable:
- dataset statistics and validation summary;
- training progress;
- training/validation metrics;
- checkpoint list;
- resume training;
- best-checkpoint selection;
- ONNX export and validation;
- beginner guidance on overfitting/underfitting.

## EPD-5 — Profiles + Replay

Add reproducible perception sessions:
- named profiles containing model, labels, camera topic, mode and runtime settings;
- import/export profile;
- replay from recorded images/rosbag where supported;
- one-click restore of known-good workcell perception configuration.

## EPD-6 — 3D Perception Tools

Strengthen manipulation-facing diagnostics:
- RGB/depth alignment checks;
- point/depth validity inspection;
- localization geometry inspector;
- tracked ID inspector;
- optional plane/background filtering tools where justified by real workcell failures.

## EPD-7 — Workcell Studio / EMD Contract

Keep repository ownership separate while formalizing the integration:
- normalized perceived-object contract;
- stable IDs and timestamps;
- pose/geometry/frame metadata;
- perception health/status;
- profile reference from a Workcell Studio scene;
- live and replay support.

EPD must not own Workcell Studio scene definitions, tasks or planning logic.

## EPD-8 — Performance Backends

After operator correctness and runtime observability are solid:
- GPU/TensorRT/Jetson paths;
- benchmark CPU vs accelerated inference;
- optional provider/backend selection;
- keep the current CPU path as a reliable fallback.

## EPD-9 — Release / Demo Quality

Produce a repeatable handoff surface:
- diagnostics bundle;
- current user guide;
- model/profile examples;
- acceptance checklist;
- screenshots/demo flow;
- known limitations;
- reproducible launch instructions.

## Priority rule

Do not jump directly to new model architectures or accelerated backends while the basic operator loop is unclear. The sequence is:

```text
camera truth
→ live feedback
→ model truth
→ training observability
→ profiles/replay
→ 3D diagnostics
→ Workcell Studio contract
→ performance backends
→ release/demo bundle
```
