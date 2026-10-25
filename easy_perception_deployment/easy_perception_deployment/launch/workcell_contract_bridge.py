#!/usr/bin/env python3
# Copyright 2026 Easy Perception Deployment contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Publish EPD P3 results as the Workcell Studio normalized perception contract.

This bridge is intentionally one-way and perception-only.  It does not read scene
geometry, bind tasks, alter PlanningScene state, call MoveIt, or command a robot.
Workcell Studio supplies only identity/provenance parameters (scene, camera and
profile reference) when it launches the bridge.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone

SNAPSHOT_SCHEMA = "workcell_perception_snapshot/v1"
STATUS_SCHEMA = "workcell_perception_status/v1"

LOCALIZATION_TOPIC = "/easy_perception_deployment/epd_localize_output"
TRACKING_TOPIC = "/easy_perception_deployment/epd_tracking_output"
DIAGNOSTICS_TOPIC = "/easy_perception_deployment/inference_diagnostics"
SNAPSHOT_TOPIC = "/workcell_studio/epd_detection_snapshot_json"
STATUS_TOPIC = "/workcell_studio/epd_connector_status"


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _stamp_ns(stamp):
    if stamp is None:
        return None
    try:
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return None
    if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        return None
    return sec * 1_000_000_000 + nanosec


def _timestamp_iso(stamp_ns):
    if stamp_ns is None:
        return ""
    sec, nanosec = divmod(int(stamp_ns), 1_000_000_000)
    moment = datetime.fromtimestamp(sec, tz=timezone.utc)
    base = moment.strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{nanosec:09d}Z"


def _point(point):
    if point is None:
        return None
    values = [
        getattr(point, "x", None),
        getattr(point, "y", None),
        getattr(point, "z", None),
    ]
    if not all(_finite(value) for value in values):
        return None
    return [float(value) for value in values]


def _quaternion(orientation):
    if orientation is None:
        return None
    values = [
        getattr(orientation, "x", None),
        getattr(orientation, "y", None),
        getattr(orientation, "z", None),
        getattr(orientation, "w", None),
    ]
    if not all(_finite(value) for value in values):
        return None
    values = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12 or abs(norm - 1.0) > 1e-3:
        return None
    return values


def _positive_dimensions(localized_object):
    values = [
        getattr(localized_object, "length", None),
        getattr(localized_object, "breadth", None),
        getattr(localized_object, "height", None),
    ]
    if not all(_finite(value) and float(value) > 0.0 for value in values):
        return None
    return [float(value) for value in values]


def _cloud_point_count(localized_object):
    cloud = getattr(localized_object, "segmented_pcl", None)
    try:
        return int(cloud.width) * int(cloud.height)
    except (AttributeError, TypeError, ValueError):
        return 0


# EPD GeometryFailure bits (epd_msgs/RejectedObject.failure_reasons).
GEOMETRY_FAILURES = (
    (1 << 0, "invalid_intrinsics"), (1 << 1, "invalid_roi"), (1 << 2, "invalid_mask"),
    (1 << 3, "insufficient_depth"), (1 << 4, "empty_cloud"), (1 << 5, "nonfinite_geometry"),
    (1 << 6, "invalid_dimensions"), (1 << 7, "invalid_orientation"), (1 << 8, "frame_mismatch"),
    (1 << 9, "geometry_exception"),
)


def _confidence(item):
    """Detector score when the EPD message reports one (older messages have none)."""
    value = getattr(item, "confidence", None)
    if value is None or not _finite(value) or not 0.0 <= float(value) <= 1.0:
        return None
    return float(value)


def _depth_support(item):
    """Measured depth coverage of the object's mask, when reported."""
    if not hasattr(item, "valid_depth_ratio"):
        return {}
    return {
        "mask_pixel_count": int(getattr(item, "mask_pixel_count", 0)),
        "valid_depth_pixel_count": int(getattr(item, "valid_depth_pixel_count", 0)),
        "valid_depth_ratio": float(getattr(item, "valid_depth_ratio", 0.0)),
    }


def _rejected_contract(rejected):
    """A detection EPD refused to localize, with EPD's own reasons (never given geometry)."""
    roi = getattr(rejected, "roi", None)
    reasons = int(getattr(rejected, "failure_reasons", 0))
    item = {
        "label": str(getattr(rejected, "name", "") or ""),
        "quality": "degraded" if int(getattr(rejected, "quality", 2)) == 1 else "invalid",
        "failure_reasons": [name for bit, name in GEOMETRY_FAILURES if reasons & bit] or ["unspecified"],
        **_depth_support(rejected),
    }
    confidence = _confidence(rejected)
    if confidence is not None:
        item["confidence"] = confidence
    if roi is not None:
        item["roi_xywh"] = [int(roi.x_offset), int(roi.y_offset), int(roi.width), int(roi.height)]
    return item


def _object_contract(
    localized_object,
    frame_id,
    source_stamp_ns,
    index,
    track_id="",
    require_tracking_ids=False,
):
    """Convert one LocalizedObject without inventing confidence or geometry."""
    errors = []
    warnings = []
    label = str(getattr(localized_object, "name", "") or "").strip()
    if not label:
        errors.append(f"objects[{index}] has no class/name label")

    stable_track_id = str(track_id or "").strip()
    if require_tracking_ids and not stable_track_id:
        errors.append(f"objects[{index}] has no stable Tracking ID")

    if stable_track_id:
        object_id = stable_track_id
        identity_scope = "tracking"
    else:
        object_id = f"localization:{source_stamp_ns}:{index}"
        identity_scope = "observation"

    centroid = _point(getattr(localized_object, "centroid", None))
    pose_msg = getattr(localized_object, "pose", None)
    pose_position = _point(getattr(pose_msg, "position", None))
    pose_quaternion = _quaternion(getattr(pose_msg, "orientation", None))

    confidence = _confidence(localized_object)
    item = {
        "object_id": object_id,
        "label": label,
        "attributes": {
            "identity_scope": identity_scope,
            "confidence_available": confidence is not None,
            "segmented_cloud_available": _cloud_point_count(localized_object) > 0,
            "segmented_cloud_points": _cloud_point_count(localized_object),
            **_depth_support(localized_object),
        },
    }
    if confidence is not None:
        item["confidence"] = confidence
    if stable_track_id:
        item["track_id"] = stable_track_id

    if pose_position is not None and pose_quaternion is not None:
        item["pose"] = {
            "frame_id": frame_id,
            "position": pose_position,
            "orientation_xyzw": pose_quaternion,
        }
    elif centroid is not None:
        item["centroid"] = centroid
        if pose_position is not None and pose_quaternion is None:
            warnings.append(
                f"objects[{index}] pose quaternion is unavailable/non-normalized; "
                "published finite centroid instead"
            )
    else:
        errors.append(f"objects[{index}] has neither a valid pose nor centroid")

    dimensions = _positive_dimensions(localized_object)
    if dimensions is not None:
        item["dimensions_xyz"] = dimensions
        item["shape"] = "box"
    else:
        warnings.append(
            f"objects[{index}] has no finite positive observed dimensions; "
            "collision geometry is unavailable"
        )

    axis = _point(getattr(localized_object, "axis", None))
    if axis is not None:
        item["attributes"]["major_axis_xyz"] = axis

    roi = getattr(localized_object, "roi", None)
    if roi is not None:
        try:
            item["attributes"]["roi_xywh"] = [
                int(roi.x_offset),
                int(roi.y_offset),
                int(roi.width),
                int(roi.height),
            ]
        except (AttributeError, TypeError, ValueError):
            pass

    return item, errors, warnings


def validate_snapshot(snapshot):
    """Validate the contract fields consumed by Workcell Studio."""
    errors = []
    if not isinstance(snapshot, dict):
        return ["snapshot must be a mapping"]
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA:
        errors.append(f"schema_version must be {SNAPSHOT_SCHEMA}")
    for key in ("scene_id", "camera_id", "timestamp", "frame_id"):
        if not snapshot.get(key):
            errors.append(f"{key} is required")

    objects = snapshot.get("objects")
    if not isinstance(objects, list):
        errors.append("objects must be a list")
        return errors

    seen = set()
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            errors.append(f"objects[{index}] must be a mapping")
            continue
        object_id = item.get("object_id") or item.get("track_id")
        if not object_id:
            errors.append(f"objects[{index}] requires object_id or track_id")
        elif str(object_id) in seen:
            errors.append(f"duplicate object id: {object_id}")
        else:
            seen.add(str(object_id))
        if not item.get("label"):
            errors.append(f"objects[{index}].label is required")

        confidence = item.get("confidence")
        if confidence is not None:
            if not _finite(confidence) or not 0.0 <= float(confidence) <= 1.0:
                errors.append(f"objects[{index}].confidence must be in [0, 1]")

        pose = item.get("pose") if isinstance(item.get("pose"), dict) else None
        centroid = item.get("centroid")
        if pose is not None:
            position = pose.get("position")
            orientation = pose.get("orientation_xyzw")
            if not (
                isinstance(position, list)
                and len(position) == 3
                and all(_finite(value) for value in position)
            ):
                errors.append(f"objects[{index}].pose.position must be finite xyz")
            if not (
                isinstance(orientation, list)
                and len(orientation) == 4
                and all(_finite(value) for value in orientation)
            ):
                errors.append(
                    f"objects[{index}].pose.orientation_xyzw must be finite xyzw"
                )
            else:
                norm = math.sqrt(sum(float(value) ** 2 for value in orientation))
                if abs(norm - 1.0) > 1e-3:
                    errors.append(
                        f"objects[{index}].pose.orientation_xyzw must be normalized"
                    )
            if pose.get("frame_id") and pose.get("frame_id") != snapshot.get("frame_id"):
                errors.append(
                    f"objects[{index}].pose.frame_id must match snapshot frame_id"
                )
        elif not (
            isinstance(centroid, list)
            and len(centroid) == 3
            and all(_finite(value) for value in centroid)
        ):
            errors.append(f"objects[{index}] requires a finite pose or centroid")

        dimensions = item.get("dimensions_xyz")
        if dimensions is not None and not (
            isinstance(dimensions, list)
            and len(dimensions) == 3
            and all(_finite(value) and float(value) > 0.0 for value in dimensions)
        ):
            errors.append(
                f"objects[{index}].dimensions_xyz must contain positive finite xyz"
            )
    return errors


def build_snapshot(
    message,
    *,
    source_kind,
    scene_id,
    camera_id,
    profile_ref="",
    runtime_mode="live",
    require_tracking_ids=False,
):
    """Normalize an EPD Localization or Tracking result into Workcell contract v1."""
    errors = []
    warnings = []
    header = getattr(message, "header", None)
    frame_id = str(getattr(header, "frame_id", "") or "").strip()
    source_stamp_ns = _stamp_ns(getattr(header, "stamp", None))

    if not str(scene_id or "").strip():
        errors.append("scene_id is required from Workcell Studio")
    if not str(camera_id or "").strip():
        errors.append("camera_id is required from Workcell Studio")
    if not frame_id:
        errors.append("EPD result header.frame_id is required")
    if source_stamp_ns is None:
        errors.append("EPD result header.stamp is invalid")

    objects = list(getattr(message, "objects", []) or [])
    tracking_ids = (
        [str(value) for value in (getattr(message, "object_ids", []) or [])]
        if source_kind == "tracking"
        else []
    )
    if source_kind == "tracking" and len(tracking_ids) != len(objects):
        warnings.append(
            "Tracking object_ids length does not match objects length; unmatched "
            "objects are observation-scoped"
        )

    normalized = []
    if source_stamp_ns is not None and frame_id:
        for index, localized_object in enumerate(objects):
            track_id = tracking_ids[index] if index < len(tracking_ids) else ""
            item, item_errors, item_warnings = _object_contract(
                localized_object,
                frame_id,
                source_stamp_ns,
                index,
                track_id=track_id,
                require_tracking_ids=require_tracking_ids,
            )
            normalized.append(item)
            errors.extend(item_errors)
            warnings.extend(item_warnings)

    topic = TRACKING_TOPIC if source_kind == "tracking" else LOCALIZATION_TOPIC
    message_type = (
        "epd_msgs/msg/EPDObjectTracking"
        if source_kind == "tracking"
        else "epd_msgs/msg/EPDObjectLocalization"
    )
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA,
        "source": "epd",
        "runtime_mode": str(runtime_mode),
        "scene_id": str(scene_id or ""),
        "camera_id": str(camera_id or ""),
        "timestamp": _timestamp_iso(source_stamp_ns),
        "timestamp_ns": source_stamp_ns,
        "frame_id": frame_id,
        "objects": normalized,
        "provenance": {
            "source_topic": topic,
            "source_message_type": message_type,
            "source_stamp_ns": source_stamp_ns,
            "process_time_ms": int(getattr(message, "process_time", 0) or 0),
            "confidence_available": bool(normalized) and all(
                "confidence" in item for item in normalized),
        },
        "rejected_objects": [
            _rejected_contract(item) for item in (getattr(message, "rejected_objects", []) or [])],
    }
    if profile_ref:
        snapshot["profile_ref"] = str(profile_ref)
    if source_kind == "tracking":
        snapshot["lost_object_ids"] = [
            str(value)
            for value in (getattr(message, "lost_track_ids", []) or [])
            if str(value)
        ]
    if warnings:
        snapshot["warnings"] = warnings

    errors.extend(validate_snapshot(snapshot))
    # Preserve first occurrence order while avoiding repeated messages.
    errors = list(dict.fromkeys(errors))
    warnings = list(dict.fromkeys(warnings))
    return snapshot, errors, warnings


def _diagnostic_level(level):
    """DiagnosticStatus.level: rclpy delivers the uint8 as a 1-byte bytes object."""
    if isinstance(level, (bytes, bytearray)):
        return level[0] if level else 0
    return int(level or 0)


def diagnostics_payload(message):
    """Extract EPD inference-worker health without changing backend semantics."""
    for status in getattr(message, "status", []) or []:
        if getattr(status, "name", "") != "easy_perception_deployment/inference_worker":
            continue
        values = {
            str(getattr(item, "key", "")): str(getattr(item, "value", ""))
            for item in getattr(status, "values", []) or []
        }
        return {
            "available": True,
            "level": _diagnostic_level(getattr(status, "level", 0)),
            "message": str(getattr(status, "message", "") or ""),
            "hardware_id": str(getattr(status, "hardware_id", "") or ""),
            "values": values,
        }
    return {"available": False, "level": None, "message": "", "values": {}}


def status_payload(
    *,
    scene_id,
    camera_id,
    profile_ref,
    runtime_mode,
    source_mode,
    last_snapshot,
    last_message_age_s,
    last_error="",
    backend=None,
    stale_timeout_s=2.0,
):
    backend = backend or {
        "available": False,
        "level": None,
        "message": "",
        "values": {},
    }
    if not scene_id or not camera_id:
        state = "FAILED"
        reason = "scene_id and camera_id must be supplied by Workcell Studio"
    elif last_error:
        state = "FAILED"
        reason = last_error
    elif backend.get("available") and int(backend.get("level") or 0) >= 2:
        state = "FAILED"
        reason = backend.get("message") or "EPD inference diagnostics report ERROR"
    elif last_snapshot is None:
        state = "WAITING"
        reason = "waiting for normalized EPD perception output"
    elif last_message_age_s is None or last_message_age_s > float(stale_timeout_s):
        state = "STALE"
        reason = "normalized EPD perception output exceeded freshness timeout"
    else:
        state = "READY"
        reason = "fresh normalized EPD perception snapshot available"

    objects = last_snapshot.get("objects", []) if isinstance(last_snapshot, dict) else []
    source = (
        last_snapshot.get("provenance", {}).get("source_message_type", "")
        if isinstance(last_snapshot, dict)
        else ""
    )
    stable_ids = bool(objects) and all(
        bool(item.get("track_id")) for item in objects if isinstance(item, dict)
    )
    return {
        "schema_version": STATUS_SCHEMA,
        "contract_schema_version": SNAPSHOT_SCHEMA,
        "mode": str(runtime_mode),
        "state": state,
        "scene_id": str(scene_id or ""),
        "camera_id": str(camera_id or ""),
        "profile_ref": str(profile_ref or ""),
        "source_mode": str(source_mode),
        "source_message_type": source,
        "last_timestamp": (
            last_snapshot.get("timestamp") if isinstance(last_snapshot, dict) else None
        ),
        "last_timestamp_ns": (
            last_snapshot.get("timestamp_ns") if isinstance(last_snapshot, dict) else None
        ),
        "object_count": len(objects) if isinstance(objects, list) else 0,
        "stable_tracking_ids": stable_ids,
        "last_msg_age_s": last_message_age_s,
        "epd_connected": state == "READY",
        "reason": reason,
        "last_error": last_error,
        "backend": backend,
    }


class WorkcellContractBridge:
    """ROS-facing publisher around the pure normalization helpers."""

    def __init__(self, node, string_type, diagnostic_status_error, args):
        self.node = node
        self.String = string_type
        self.diagnostic_status_error = diagnostic_status_error
        self.args = args
        self.last_snapshot = None
        self.last_wall_time = None
        self.last_source_stamp_ns = None
        self.last_error = ""
        self.backend = {
            "available": False,
            "level": None,
            "message": "",
            "values": {},
        }

        self.snapshot_pub = node.create_publisher(
            string_type,
            args.snapshot_topic,
            10,
        )
        self.status_pub = node.create_publisher(
            string_type,
            args.status_topic,
            10,
        )

    def handle_result(self, message, source_kind):
        if self.args.source_mode not in ("auto", source_kind):
            return
        source_stamp_ns = _stamp_ns(getattr(getattr(message, "header", None), "stamp", None))
        if source_stamp_ns is not None and self.last_source_stamp_ns is not None:
            if source_stamp_ns < self.last_source_stamp_ns:
                self.last_error = (
                    "EPD source timestamp moved backward; normalized snapshot rejected"
                )
                return
            if source_stamp_ns == self.last_source_stamp_ns:
                # In auto mode, allow Tracking to replace same-stamp Localization.
                previous_type = (
                    self.last_snapshot.get("provenance", {}).get("source_message_type", "")
                    if isinstance(self.last_snapshot, dict)
                    else ""
                )
                if not (
                    source_kind == "tracking"
                    and previous_type == "epd_msgs/msg/EPDObjectLocalization"
                ):
                    return

        snapshot, errors, _warnings = build_snapshot(
            message,
            source_kind=source_kind,
            scene_id=self.args.scene_id,
            camera_id=self.args.camera_id,
            profile_ref=self.args.profile_ref,
            runtime_mode=self.args.runtime_mode,
            require_tracking_ids=self.args.require_tracking_ids,
        )
        if errors:
            self.last_error = "; ".join(errors)
            return

        payload = self.String()
        payload.data = json.dumps(snapshot, separators=(",", ":"), sort_keys=True)
        self.snapshot_pub.publish(payload)
        self.last_snapshot = snapshot
        self.last_wall_time = time.monotonic()
        self.last_source_stamp_ns = source_stamp_ns
        self.last_error = ""

    def handle_diagnostics(self, message):
        self.backend = diagnostics_payload(message)

    def publish_status(self):
        age = None
        if self.last_wall_time is not None:
            age = max(0.0, time.monotonic() - self.last_wall_time)
        payload = status_payload(
            scene_id=self.args.scene_id,
            camera_id=self.args.camera_id,
            profile_ref=self.args.profile_ref,
            runtime_mode=self.args.runtime_mode,
            source_mode=self.args.source_mode,
            last_snapshot=self.last_snapshot,
            last_message_age_s=age,
            last_error=self.last_error,
            backend=self.backend,
            stale_timeout_s=self.args.stale_timeout_s,
        )
        message = self.String()
        message.data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        self.status_pub.publish(message)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--profile-ref", default="")
    parser.add_argument("--runtime-mode", choices=("live", "replay"), default="live")
    parser.add_argument(
        "--source-mode",
        choices=("tracking", "localization", "auto"),
        default="tracking",
    )
    parser.add_argument("--localization-topic", default=LOCALIZATION_TOPIC)
    parser.add_argument("--tracking-topic", default=TRACKING_TOPIC)
    parser.add_argument("--diagnostics-topic", default=DIAGNOSTICS_TOPIC)
    parser.add_argument("--snapshot-topic", default=SNAPSHOT_TOPIC)
    parser.add_argument("--status-topic", default=STATUS_TOPIC)
    parser.add_argument("--stale-timeout-s", type=float, default=2.0)
    parser.add_argument("--status-period-s", type=float, default=0.25)
    parser.add_argument("--require-tracking-ids", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.stale_timeout_s <= 0.0:
        raise SystemExit("--stale-timeout-s must be positive")
    if args.status_period_s <= 0.0:
        raise SystemExit("--status-period-s must be positive")

    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
    from epd_msgs.msg import EPDObjectLocalization, EPDObjectTracking
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    rclpy.init()
    node = Node("epd_workcell_contract_bridge")
    bridge = WorkcellContractBridge(node, String, DiagnosticStatus.ERROR, args)

    # Latest result only, delivered whole: results carry the depth image, masks and
    # clouds, and best-effort readers lost most of them to fragment loss.
    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.RELIABLE
    subscriptions = []
    if args.source_mode in ("localization", "auto"):
        subscriptions.append(
            node.create_subscription(
                EPDObjectLocalization,
                args.localization_topic,
                lambda message: bridge.handle_result(message, "localization"),
                qos,
            )
        )
    if args.source_mode in ("tracking", "auto"):
        subscriptions.append(
            node.create_subscription(
                EPDObjectTracking,
                args.tracking_topic,
                lambda message: bridge.handle_result(message, "tracking"),
                qos,
            )
        )
    subscriptions.append(
        node.create_subscription(
            DiagnosticArray,
            args.diagnostics_topic,
            bridge.handle_diagnostics,
            10,
        )
    )
    timer = node.create_timer(args.status_period_s, bridge.publish_status)

    node.get_logger().info(
        "Workcell contract bridge ready: scene=%s camera=%s profile=%s "
        "mode=%s source=%s snapshot=%s status=%s"
        % (
            args.scene_id,
            args.camera_id,
            args.profile_ref or "<none>",
            args.runtime_mode,
            args.source_mode,
            args.snapshot_topic,
            args.status_topic,
        )
    )
    try:
        rclpy.spin(node)
    finally:
        del timer
        del subscriptions
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
