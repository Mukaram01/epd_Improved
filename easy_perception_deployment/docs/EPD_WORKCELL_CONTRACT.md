# EPD-7 — Workcell Studio / EMD Contract

EPD-7 formalizes the perception boundary between Easy Perception Deployment (EPD) and Workcell Studio / Easy Manipulator (EMD).

The boundary is deliberately one-way:

```text
RealSense / replay
        ↓
       EPD
        ↓
workcell_perception_snapshot/v1
workcell_perception_status/v1
        ↓
Workcell Studio / EMD
        ↓
scene identity validation → task binding → PlanningScene → grasp → MoveIt
```

EPD owns camera processing, inference, localization, tracking and observed geometry. Workcell Studio / EMD owns scene definitions, task intent, PlanningScene conversion, grasp selection and motion planning.

EPD-7 does **not** add robot motion, PlanningScene writes, task selection or scene authoring to EPD.

## Why this bridge exists

Native EPD P3 messages are useful ROS interfaces, but they do not carry Workcell Studio scene identity or an EPD-5 profile reference. Workcell Studio already validates a normalized `workcell_perception_snapshot/v1` payload before task binding.

EPD-7 adds a small read-only bridge that converts the existing native Localization/Tracking results into that normalized contract while preserving source truth.

The bridge is installed with the package because the complete `launch/` directory is already installed. It is started through `workcell_contract.launch.py`; it is not a second perception engine.

## Normalized snapshot topic

Default topic:

```text
/workcell_studio/epd_detection_snapshot_json
```

Message type:

```text
std_msgs/msg/String
```

The string contains JSON with schema:

```text
workcell_perception_snapshot/v1
```

Example:

```json
{
  "schema_version": "workcell_perception_snapshot/v1",
  "source": "epd",
  "runtime_mode": "live",
  "scene_id": "ur5_2f_test",
  "camera_id": "realsense_d435i_1",
  "profile_ref": "D435i Table Pick",
  "timestamp": "2026-09-06T12:10:15.123456789Z",
  "timestamp_ns": 1788696615123456789,
  "frame_id": "camera_color_optical_frame",
  "objects": [
    {
      "object_id": "42",
      "track_id": "42",
      "label": "part",
      "pose": {
        "frame_id": "camera_color_optical_frame",
        "position": [0.11, -0.04, 0.62],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]
      },
      "dimensions_xyz": [0.08, 0.04, 0.03],
      "shape": "box",
      "attributes": {
        "identity_scope": "tracking",
        "confidence_available": false,
        "segmented_cloud_available": true,
        "segmented_cloud_points": 1250
      }
    }
  ],
  "lost_object_ids": ["17"],
  "provenance": {
    "source_topic": "/easy_perception_deployment/epd_tracking_output",
    "source_message_type": "epd_msgs/msg/EPDObjectTracking",
    "source_stamp_ns": 1788696615123456789,
    "process_time_ms": 38,
    "confidence_available": false
  }
}
```

## Stable identity

Tracking is the recommended source for Workcell Studio.

With Tracking:

- `object_id` is the exact EPD stable tracking ID;
- `track_id` contains the same backend tracking ID;
- `lost_object_ids` is copied from EPD `lost_track_ids`;
- the GUI/bridge does not synthesize replacement stable IDs.

`workcell_contract.launch.py` therefore defaults to:

```text
source_mode:=tracking
require_tracking_ids:=true
```

Localization-only output can still be normalized by setting:

```text
source_mode:=localization require_tracking_ids:=false
```

Localization has no backend stable ID. In that mode EPD-7 creates an explicitly **observation-scoped** identifier:

```text
localization:<source_stamp_ns>:<object_index>
```

The object includes:

```json
"attributes": {"identity_scope": "observation"}
```

That ID must not be treated as persistent tracking identity by downstream code.

## Timestamp truth

The normalized timestamp comes directly from the native EPD result header.

EPD-7 publishes both:

- `timestamp`: UTC ISO-8601 text;
- `timestamp_ns`: the exact ROS source timestamp in nanoseconds.

The bridge rejects source timestamps that move backward. Duplicate same-stamp results are not republished, except that an `auto` source may replace a same-stamp Localization result with the stronger Tracking result.

No wall-clock timestamp is substituted for the sensor/result timestamp.

## Pose, frame and observed geometry

EPD-7 uses only fields already present in `LocalizedObject`:

- `pose.position`;
- `pose.orientation`;
- `centroid`;
- `length`, `breadth`, `height`;
- `segmented_pcl`;
- `axis`;
- ROI;
- class/name.

A pose is published only when its position is finite and its quaternion is finite and normalized. If the quaternion is invalid but the centroid is valid, the bridge publishes the centroid instead and adds a warning. It does **not** invent an identity quaternion.

Positive finite observed `length/breadth/height` become:

```json
"dimensions_xyz": [length, breadth, height],
"shape": "box"
```

This matches Workcell Studio's current dynamic PlanningScene box bridge. If observed dimensions are unavailable, they are omitted and the snapshot warning says collision geometry is unavailable. EPD-7 never substitutes authored or guessed dimensions.

The snapshot `frame_id` is the native EPD result header frame. Object pose `frame_id` must match it. TF conversion remains a downstream Workcell Studio / MoveIt responsibility.

## Confidence

`EPDObjectLocalization` and `EPDObjectTracking` do not currently carry a per-object confidence field. EPD-7 therefore does **not** invent one.

Normalized objects omit `confidence` and explicitly include:

```json
"attributes": {"confidence_available": false}
```

The snapshot provenance repeats `confidence_available: false`.

If the native P3 contract is extended later with a trustworthy object-confidence association, EPD-7 can populate the optional normalized field without changing the Workcell snapshot schema.

## Profile reference

Workcell Studio owns the scene and chooses which EPD-5 perception profile belongs to that scene. EPD does not scan or edit the scene file.

The scene/launcher passes the profile reference into EPD-7:

```text
profile_ref:=D435i Table Pick
```

The bridge copies it into snapshot and status provenance. This is a reference only; the bridge does not apply or mutate an EPD-5 profile.

## Health/status topic

Default topic:

```text
/workcell_studio/epd_connector_status
```

Message type:

```text
std_msgs/msg/String
```

Schema:

```text
workcell_perception_status/v1
```

States are intentionally small and compatible with the current Workcell Studio adapter vocabulary:

- `WAITING` — configured but no normalized result has arrived;
- `READY` — a fresh valid normalized snapshot is available;
- `STALE` — the last valid normalized snapshot exceeded the freshness timeout;
- `FAILED` — missing required scene/camera identity, contract conversion failure, timestamp regression, or backend diagnostics at ERROR level.

Status also contains:

- scene and camera identity;
- profile reference;
- live/replay mode;
- source mode;
- last source timestamp;
- object count;
- whether all current objects have stable Tracking IDs;
- local message age;
- EPD inference-worker diagnostic level/message and raw key/value counters.

The bridge does not reinterpret cumulative backend counters as a current failure. Current failure state follows the ROS diagnostic status level or an actual bridge validation failure.

## Live launch

The existing EPD/EMD pipeline now exposes optional normalized-contract launch arguments. Existing launch behavior remains unchanged unless the bridge is enabled.

Recommended Workcell Studio launch:

```bash
ros2 launch easy_perception_deployment epd_emd_pipeline.launch.py \
  usecase_mode_override:=4 \
  publish_workcell_contract:=true \
  workcell_scene_id:=ur5_2f_test \
  workcell_camera_id:=realsense_d435i_1 \
  perception_profile_ref:="D435i Table Pick" \
  contract_source_mode:=tracking \
  contract_require_tracking_ids:=true
```

Then inspect:

```bash
ros2 topic echo /workcell_studio/epd_detection_snapshot_json --once
ros2 topic echo /workcell_studio/epd_connector_status --once
```

## Standalone contract launch

If EPD is already running, start only the adapter:

```bash
ros2 launch easy_perception_deployment workcell_contract.launch.py \
  scene_id:=ur5_2f_test \
  camera_id:=realsense_d435i_1 \
  profile_ref:="D435i Table Pick" \
  source_mode:=tracking \
  require_tracking_ids:=true
```

## Replay support

`replay.launch.py` can publish the same normalized contract from the existing deterministic P8 replay path:

```bash
ros2 launch easy_perception_deployment replay.launch.py \
  publish_workcell_contract:=true \
  workcell_scene_id:=ur5_2f_test \
  workcell_camera_id:=fixture_camera \
  perception_profile_ref:="P8 acceptance replay"
```

The snapshot declares:

```json
"runtime_mode": "replay"
```

The object schema, stable Tracking IDs, timestamps and geometry fields are otherwise the same as live mode. This lets Workcell Studio exercise the same contract against deterministic replay without pretending replay is live hardware.

## Migration from the older Workcell-side converter

Easy Manipulator / Workcell Studio already contains `epd_to_workcell_snapshot_node.py`, which can normalize JSON-like EPD payloads and publish the same downstream snapshot/status topic names.

When EPD-7's direct bridge is enabled, **do not run two publishers for the same normalized snapshot/status topics**. Use one of these paths:

1. **EPD-7 direct contract (recommended for the current EPD fork)** — EPD converts native P3 ROS messages and publishes `workcell_perception_snapshot/v1` directly.
2. **Legacy Workcell-side adapter** — leave EPD-7 disabled and let Workcell Studio perform the conversion.

Both preserve the same ownership boundary; the direct EPD-7 path removes the ambiguous native-message-to-JSON conversion step.

## Machine-readable schema

Repository schemas:

- `docs/schemas/workcell_perception_snapshot_v1.schema.json`
- `docs/schemas/workcell_perception_status_v1.schema.json`

The runtime bridge also performs a lightweight validation before publishing. Invalid normalized snapshots are rejected and surfaced through `workcell_perception_status/v1` instead of being sent downstream.

## Manual acceptance

1. Start a P3 Tracking deployment with a RealSense camera.
2. Start the contract bridge with explicit scene ID, camera ID and profile reference.
3. Confirm status moves `WAITING → READY` after the first valid Tracking result.
4. Confirm `object_id == track_id` and remains stable while the same object persists.
5. Remove the tracked object and confirm backend `lost_track_ids` appears as `lost_object_ids`.
6. Confirm source `timestamp_ns` and frame ID match the native EPD Tracking message.
7. Confirm pose/centroid and observed box dimensions match the native `LocalizedObject` values.
8. Confirm no confidence value is fabricated.
9. Stop the camera or EPD and confirm status moves to `STALE` after the configured timeout.
10. Run deterministic replay with contract publishing enabled and confirm the same snapshot schema is produced with `runtime_mode: replay`.
11. Confirm no Workcell scene/task/PlanningScene/MoveIt/robot-motion operation exists in the bridge.
