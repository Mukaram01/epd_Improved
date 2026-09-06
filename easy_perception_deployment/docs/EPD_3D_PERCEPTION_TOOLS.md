# EPD-6 — 3D Perception Tools

EPD-6 adds a read-only 3D diagnostics surface for the existing P3 Localization and Tracking pipeline. It does not change inference, scene ownership, planning, filtering, or robot-motion behavior.

## Open the inspector

From **Deploy**, select **3D Inspector** or press `Ctrl+Shift+3`.

The inspector listens to the production EPD topics:

- `/easy_perception_deployment/epd_localization_output`
- `/easy_perception_deployment/epd_tracking_output`
- `/easy_perception_deployment/inference_diagnostics`

Use a P3-compatible model and select **Localization** or **Tracking** to receive 3D object data.

## 3D health

The health tab checks only facts present in the current P3 result:

- result frame width/height;
- embedded depth-image width/height;
- depth encoding (`16UC1` or `32FC1`);
- finite, positive `fx` and `fy` plus finite `ppx`/`ppy`;
- exact result-header versus embedded-depth source timestamp;
- sampled valid-depth ratio;
- frame ID;
- processing time.

A result is labelled **ALIGNED** only when dimensions, intrinsics, depth encoding and source timestamp all agree. Camera Assistant remains the independent tool for checking live RGB, aligned-depth and CameraInfo publishers.

### Sampled depth ratio

The GUI samples at most about 2,048 pixels from the embedded depth image. Positive finite depth is considered usable. This is deliberately labelled as a sample estimate rather than a full-frame measurement.

## Localized objects

For every `LocalizedObject`, the inspector displays:

- Tracking ID when available;
- class/name;
- centroid X/Y/Z in the camera frame;
- length, breadth and height;
- segmented point-cloud point count;
- major-axis vector magnitude;
- a GUI-side inspector check.

The inspector check is conservative:

- **VALID** — finite positive-Z centroid, positive dimensions, non-empty point cloud, usable axis and pose quaternion;
- **DEGRADED** — centroid is usable but one or more supporting geometry fields are missing/weak;
- **INVALID** — centroid is non-finite or not in front of the camera.

This is an operator sanity check only. The production inference worker remains the source of truth for geometry quality counters.

## Tracking IDs

In Tracking mode, EPD-6 shows:

- current stable `object_ids` from the latest observation;
- `lost_track_ids` reported by the production tracking lifecycle.

No IDs are invented or reconstructed by the GUI.

## Production geometry diagnostics

The inspector reads the existing `easy_perception_deployment/inference_worker` diagnostic status and exposes the counters already produced by EPD, including:

- `detections_total`
- `geometry_valid_total`
- `geometry_degraded_total`
- `geometry_invalid_total`
- `invalid_intrinsics_total`
- `empty_mask_total`
- `insufficient_depth_total`
- `empty_cloud_total`
- `nonfinite_geometry_total`

Where available it also shows tracking/observation counters such as confirmed IDs, tracks created/lost and the latest completed observation identity.

## Plane/background filtering

EPD-6 intentionally does **not** switch on plane or background filtering. The roadmap described filtering as optional and evidence-driven. The new diagnostics should first show a repeatable real-workcell failure, for example table points contaminating geometry or depth holes dominating an object. A later filter should then target that measured failure instead of changing perception behavior pre-emptively.

## Stale and unavailable states

The inspector never leaves old data labelled live indefinitely. If no fresh P3 result arrives for roughly three seconds, the UI returns to a waiting state. If ROS 2 Python support or EPD messages are unavailable, the inspector reports **UNAVAILABLE** rather than pretending the pipeline is healthy.

## Safety and ownership

EPD-6 is diagnostic only:

- no model/inference changes;
- no ROS message-schema changes;
- no filter changes;
- no PlanningScene changes;
- no EMD/Workcell Studio task ownership changes;
- no MoveIt or robot-motion commands.

This keeps EPD responsible for perception while downstream Workcell Studio/EMD remains responsible for scene, task, grasp and motion-planning behavior.

## Manual acceptance

1. Start a RealSense-compatible P3 Localization or Tracking deployment.
2. Open **3D Inspector**.
3. Confirm frame/depth shape and source timestamps show aligned when the pipeline is healthy.
4. Confirm sampled valid-depth percentage is plausible for the scene.
5. Place an object in view and confirm centroid, dimensions and point-cloud count update.
6. In Tracking mode, confirm stable IDs persist across frames.
7. Remove a tracked object and confirm its ID appears in the LOST transition output when the backend reports it.
8. Stop the camera or deployment and confirm the inspector becomes waiting/stale instead of remaining LIVE.
9. Review production geometry counters and confirm no GUI-side number is presented as a backend counter.
