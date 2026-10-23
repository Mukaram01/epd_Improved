# Perception Monitor

The Perception Monitor is the Deploy workspace for answering four questions without a camera:

1. What does the perception system see?
2. What did it detect?
3. Which detections have usable 3D information?
4. Why was a detection degraded or rejected?

It is observability only. It does not change inference, thresholds, RGB-D geometry gates, tracking, ROS or EMD contracts, Workcell Studio or robot motion.

## Workspace

Deploy is organised preview-first:

| Area | Contents |
|---|---|
| Top bar | **Live camera** / **Replay recording**, replay transport, position, health badge |
| Left, top | Live mode: the existing Live Perception card. Replay mode: the recorded frame with detection boxes |
| Left, bottom | **Detections**: one row per detector proposal from the readiness report |
| Right tabs | **Inspector** (selected detection), **Health**, **Setup** (the existing model, perception, camera, runtime and readiness cards) |
| Footer | Deployment status and **Run Perception** |

The panels are splitters, so they can be resized. The layout is checked at 1366×768.

## Readiness states

The table shows the states that `EPD::makeReadinessReport` publishes on `/easy_perception_deployment/detection_readiness` (`diagnostic_msgs/DiagnosticArray`, one status per proposal, stamped with the source RGB header):

| State | Meaning |
|---|---|
| `REJECTED` | 2D evidence unusable or contested, for example overlapping ownership. Class and confidence are still reported. |
| `DETECTED` | Class, confidence and an owned 2D mask, but no validated 3D position. |
| `LOCALISED` | Measured 3D position; the orientation is not valid. |
| `POSE_VALID` | Full pose, but not published; the note says why, for example `not_selected`, `max_detections_cap` or `no_track_assignment`. |
| `GRASP_CANDIDATE` | Published on the valid-pose output for downstream planning. |

`GRASP_CANDIDATE` is a perception statement only. It does not mean the object is reachable, collision-free or graspable; that is decided downstream.

Columns: proposal index, class, confidence, track ID, state, reason, measured centroid, source stamp and observation age. Reasons are the emitted failure names (for example `overlapping_ownership` or `ambiguous_axis_no_pose`), shown in plain language with the raw code in the Inspector. The GUI never derives a state, reason or pose of its own. Geometry beyond what a state validates is not displayed.

Missing values are shown as *Unavailable*, never as zero. Readiness reports do not carry object dimensions, mask images or latency, so those fields always read *Unavailable*.

## Image and detection association

A detection box is drawn only when the RGB frame and the readiness report have the identical source stamp and frame ID. Being the latest message is never enough. If they differ, the frame is shown without boxes, and the reason is printed on the image and in the Inspector.

In live mode the preview frame is not matched to reports by stamp, so live boxes are not drawn. The EPD `image_output` overlay remains available in the Live Perception card.

## Replay without a camera

Choose **Replay recording**, then **Open recording…** and select a folder containing `session.json`. The recording is checked before playback:

- schema and provenance (`recorded` or `synthetic`) are required;
- image paths must stay inside the recording;
- each frame's SHA256 is checked when the frame is first shown (one read for both the hash and the image), so long recordings open without freezing the GUI. A modified frame is never displayed: replay stops and Health shows `FAILED`.

Playback follows the original source-stamp spacing. Gaps longer than 1.5 s, for example between separately captured scenes, are shortened and reported in Health; the stamps themselves are unchanged. **Play**, **Pause**, **Restart**, **Stop** and the frame-step buttons (◀ ▶, while paused) are available. When the recording ends, the last frame stays visible but is dimmed and marked `REPLAY ENDED · NOT UPDATING`, and the health state is `STALE`. **Stop** clears the detections.

Replay is always labelled `RECORDED REPLAY · NOT A LIVE CAMERA` or `SYNTHETIC TEST FIXTURE · NOT CAMERA DATA`.

### Making a recording

Record EPD in hybrid RGB-D mode (Localization or Tracking with `hybrid_refinement_enabled:=true`), then export:

```bash
ros2 bag record -o run1 /camera/camera/color/image_raw \
  /easy_perception_deployment/detection_readiness
ros2 run easy_perception_deployment epd_perception_session_export.py \
  --bag run1 --output run1_session --image-topic /camera/camera/color/image_raw
```

The exporter copies readiness messages losslessly, pairs each report with the RGB frame that has the identical header, and keeps a report whose frame is missing as *RGB frame missing*. It refuses to overwrite a non-empty output folder.

### Bundled test fixtures

`fixtures/perception_sessions/` contains **synthetic** fixtures: drawn shapes and hand-written reports in the producer's exact format. They exercise all five states, a missing report, a deliberately mismatched pairing, a missing RGB frame, an empty report and a recording without the readiness topic. They are not measurements. `make_synthetic_sessions.py` regenerates them.

## Health

| State | When |
|---|---|
| `STOPPED` | Nothing is running, or replay is stopped or not yet started |
| `STARTING` | Deployment is starting |
| `LIVE` | Readiness reports are arriving on the topic |
| `REPLAY ACTIVE` | A recording is playing or paused |
| `WAITING` | Deployment is running but no report has arrived yet |
| `STALE` | The last report is too old (at least 5 s and at least three typical report periods), or replay has ended |
| `FAILED` | Deployment failed, or the recording could not be loaded |
| `UNAVAILABLE` | ROS Python is unavailable, or the readiness topic has no publisher 5 s after start |

The readiness topic is published only with hybrid RGB-D refinement. Without it, Deploy shows **READINESS UNAVAILABLE**, not an empty table. An empty table appears only when a report actually contains zero proposals.

## Acceptance without hardware

```bash
cd easy_perception_deployment/gui
python3 -m pytest test_perception_monitor.py
# Full application run (use a copy of the package; Deploy writes config files):
python3 perception_monitor_acceptance.py --session <recording> --out <dir>
```

The acceptance script starts the real application, drives replay, inspection, the failure cases, 20 start/stop cycles and shutdown, and saves screenshots grabbed from the running window. The process exit code is the verdict. With ROS 2 sourced (use an isolated `ROS_DOMAIN_ID`), it also publishes the recording's readiness messages on the production topic to check the live consumer and stale handling.

Live-camera operation, GPU performance and perception accuracy are outside this acceptance.
