# EPD-2 — Live Perception View

EPD-2 embeds the normal camera/perception preview directly in **Deploy**. The operator should not need `rqt_image_view` for routine setup and validation.

## What the view shows

The Live Perception card contains:

- the current camera or EPD visualization image;
- the actual preview source;
- Detection overlay state;
- Object masks state;
- object count when the selected perception output supplies it;
- pipeline FPS when mode output messages are available;
- latency when EPD `process_time` or a comparable ROS timestamp is available;
- age of the most recently rendered frame;
- an explicit runtime state.

## Preview source rules

### Perception stopped

The card subscribes to the RGB topic configured in Deploy.

Example:

```text
/camera/camera/color/image_raw
```

This lets the operator frame the workcell and confirm that the selected camera topic is useful before starting inference.

### Perception running + Detection overlay On

The card switches to EPD visualization output:

```text
/easy_perception_deployment/image_output
```

The existing EPD runtime remains responsible for drawing supported boxes/masks. The GUI does not invent detections or alter inference results.

### Perception running + Detection overlay Off

The card keeps showing the selected camera RGB stream. ROS perception results continue to publish even though the human visualization overlay is disabled.

## Image transport

The preview follows the Deploy transport setting.

### raw

Supported raw encodings:

```text
rgb8
bgr8
rgba8
bgra8
mono8
```

The GUI converts these directly with Qt and does not add a `cv_bridge` dependency.

### compressed

For compressed transport the preview subscribes to the corresponding `/compressed` topic and uses Qt image decoding.

## Object count

EPD-2 uses the active perception output rather than counting graphics on screen.

- Classification / Counting use the detection output `class_indices` length.
- Color-Matching uses the P3 detection output `class_indices` length.
- Localization uses the localization output `objects` length.
- Tracking uses the tracking output `objects` length.

A missing output is shown as `—`, not as zero.

## FPS and latency

When perception is active, FPS is derived from mode output message arrival times when available. Before perception starts, the preview can show the camera-frame rate seen by the GUI.

Latency is taken from EPD `process_time` when the message provides it. Otherwise EPD-2 only uses ROS header age when the ROS timestamp is reasonably comparable with wall time. If neither is trustworthy, the GUI shows `—`.

## Runtime states

The preview uses explicit states:

- **STOPPED** — perception is not running;
- **STARTING** — deployment has been requested and is starting;
- **LIVE** — perception is running and fresh preview frames are arriving;
- **WAITING** — perception is running but no fresh preview frame is available;
- **STOPPING** — deployment is shutting down;
- **FAILED** — the deployment job failed;
- **UNAVAILABLE** — the GUI cannot create the ROS image subscriber.

A stale frame is never left labelled **LIVE**.

## ROS Python requirement

The embedded view uses `rclpy` plus `sensor_msgs` in a background thread. If the card reports that ROS Python image support is unavailable, launch EPD from a shell where ROS 2 and the EPD workspace are sourced.

Typical Humble setup:

```bash
source /opt/ros/humble/setup.bash
source ~/epd_ros2_ws/install/setup.bash
```

Then relaunch the GUI.

## Useful troubleshooting commands

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /easy_perception_deployment/image_output
ros2 topic echo /easy_perception_deployment/epd_tracking_output --once
```

Use **Camera Assistant** when the RGB/depth/CameraInfo health itself is uncertain.

## Safety and ownership

EPD-2 is an observability/UI increment only.

It does not:

- change inference algorithms;
- change EPD ROS message schemas;
- silently rewrite the selected camera topic;
- own Workcell Studio scenes/tasks;
- change robot planning or motion.
