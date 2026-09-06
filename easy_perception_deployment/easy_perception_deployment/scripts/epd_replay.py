#!/usr/bin/env python3
"""Deterministic sensor-observation replay and compact P8 acceptance summary."""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray
from epd_msgs.msg import EPDObjectTracking
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


class Replay(Node):
    def __init__(self):
        super().__init__("epd_replay")
        self.declare_parameter("fixture", "")
        self.declare_parameter("mode", "fast")
        self.declare_parameter("summary_output", "/tmp/epd_replay_summary.json")
        fixture_path = Path(self.get_parameter("fixture").value).resolve()
        self.mode = self.get_parameter("mode").value
        self.summary_path = Path(self.get_parameter("summary_output").value)
        with fixture_path.open(encoding="utf-8") as stream:
            self.fixture = json.load(stream)
        if self.fixture.get("schema_version") != 1:
            raise ValueError("unsupported replay fixture schema_version")
        self.fixture_dir = fixture_path.parent
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)
        self.rgb_pub = self.create_publisher(
            Image, "/easy_perception_deployment/ingress/color/image_raw", qos)
        self.depth_pub = self.create_publisher(
            Image, "/easy_perception_deployment/ingress/aligned_depth/image_raw", qos)
        self.info_pub = self.create_publisher(
            CameraInfo, "/easy_perception_deployment/ingress/color/camera_info", qos)
        self.diagnostics_sub = self.create_subscription(
            DiagnosticArray, "/easy_perception_deployment/inference_diagnostics",
            self._diagnostics, 10)
        self.tracking_sub = self.create_subscription(
            EPDObjectTracking, "/easy_perception_deployment/epd_tracking_output",
            self._tracking, QoSProfile(
                history=HistoryPolicy.KEEP_LAST, depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE))
        self.bridge = CvBridge()
        self.metrics = {}
        self.seen_ids = set()
        self.id_updates = {}
        self.lost_id_events = []

    def _diagnostics(self, message):
        for status in message.status:
            if status.name == "easy_perception_deployment/inference_worker":
                self.metrics = {item.key: item.value for item in status.values}

    def _tracking(self, message):
        for object_id in message.object_ids:
            self.id_updates[object_id] = self.id_updates.get(object_id, 0) + 1
            self.seen_ids.add(object_id)
        if message.lost_track_ids:
            self.lost_id_events.append({
                "source_stamp_ns": message.header.stamp.sec * 1_000_000_000 +
                message.header.stamp.nanosec,
                "lost_track_ids": list(message.lost_track_ids),
            })

    def _spin_until(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if predicate():
                return True
        return False

    @staticmethod
    def _number(metrics, key):
        try:
            return int(float(metrics.get(key, "0")))
        except ValueError:
            return 0

    @staticmethod
    def _float_number(metrics, key):
        try:
            return float(metrics.get(key, "0"))
        except ValueError:
            return 0.0

    def _load_rgb(self, spec, width, height):
        if "rgb" not in spec:
            return np.zeros((height, width, 3), dtype=np.uint8)
        image = cv2.imread(
            str((self.fixture_dir / spec["rgb"]).resolve()),
            cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("unable to read RGB fixture asset: " + spec["rgb"])
        if (spec.get("crop_center") and image.shape[0] >= height and
                image.shape[1] >= width):
            top = (image.shape[0] - height) // 2
            left = (image.shape[1] - width) // 2
            return image[top:top + height, left:left + width].copy()
        if image.shape[:2] != (height, width):
            return cv2.resize(image, (width, height))
        return image

    def _load_depth(self, spec, width, height):
        if "depth" in spec:
            depth = cv2.imread(
                str((self.fixture_dir / spec["depth"]).resolve()),
                cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError(
                    "unable to read depth fixture asset: " + spec["depth"])
            if (spec.get("crop_center") and depth.shape[0] >= height and
                    depth.shape[1] >= width):
                top = (depth.shape[0] - height) // 2
                left = (depth.shape[1] - width) // 2
                depth = depth[top:top + height, left:left + width].copy()
            elif depth.shape[:2] != (height, width):
                depth = cv2.resize(
                    depth, (width, height), interpolation=cv2.INTER_NEAREST)
            return depth.astype(np.uint16)
        depth = np.full(
            (height, width), int(spec.get("depth_mm", 0)), dtype=np.uint16)
        for region in spec.get("depth_regions", []):
            x = int(region["x"])
            y = int(region["y"])
            region_width = int(region["width"])
            region_height = int(region["height"])
            if (x < 0 or y < 0 or x + region_width > width or
                    y + region_height > height):
                raise ValueError("depth region lies outside the fixture image")
            depth[y:y + region_height, x:x + region_width] = int(
                region["depth_mm"])
        return depth

    def _messages(self, spec):
        width = int(self.fixture["camera"]["width"])
        height = int(self.fixture["camera"]["height"])
        stamp_ns = int(spec["stamp_ns"])
        frame = self.fixture["camera"]["frame_id"]
        rgb = self.bridge.cv2_to_imgmsg(
            self._load_rgb(spec, width, height), encoding="bgr8")
        depth = self.bridge.cv2_to_imgmsg(
            self._load_depth(spec, width, height), encoding="16UC1")
        for message in (rgb, depth):
            message.header.stamp.sec = stamp_ns // 1_000_000_000
            message.header.stamp.nanosec = stamp_ns % 1_000_000_000
            message.header.frame_id = frame
        info = CameraInfo()
        info.header = rgb.header
        info.width, info.height = width, height
        info.k = [float(value) for value in self.fixture["camera"]["k"]]
        return rgb, depth, info

    def run(self):
        if self.mode not in ("fast", "realtime"):
            raise ValueError("mode must be 'fast' or 'realtime'")
        if not self._spin_until(
                lambda: self.rgb_pub.get_subscription_count() > 0 and
                self.depth_pub.get_subscription_count() > 0 and
                self.info_pub.get_subscription_count() > 0 and
                self.count_publishers(
                    "/easy_perception_deployment/epd_tracking_output") > 0,
                20.0):
            return self._finish(
                False, "production subscribers did not become ready")
        self._spin_until(lambda: False, 1.0)
        observations = self.fixture["observations"]
        first_stamp = int(observations[0]["stamp_ns"])
        duplicate_or_regressed = 0
        previous_stamp = None
        started = time.monotonic()
        accepted = 0
        for spec in observations:
            stamp = int(spec["stamp_ns"])
            if previous_stamp is not None and stamp <= previous_stamp:
                duplicate_or_regressed += 1
            previous_stamp = stamp
            if self.mode == "realtime":
                target = (stamp - first_stamp) / 1e9
                while time.monotonic() - started < target:
                    rclpy.spin_once(self, timeout_sec=0.01)
            rgb, depth, info = self._messages(spec)
            baseline_id = self._number(self.metrics, "latest_observation_id")
            delivered = False
            for _attempt in range(5):
                self.rgb_pub.publish(rgb)
                self.depth_pub.publish(depth)
                self.info_pub.publish(info)
                if self._spin_until(
                        lambda: self._number(
                            self.metrics, "latest_observation_id") > baseline_id,
                        3.0):
                    delivered = True
                    break
            if not delivered:
                return self._finish(
                    False, "observation was not accepted",
                    duplicate_or_regressed)
            accepted = self._number(self.metrics, "latest_observation_id")
            if spec.get("wait_for_completion", True):
                if not self._spin_until(
                        lambda: self._number(
                            self.metrics,
                            "last_completed_observation_id") >= accepted or
                        self._number(self.metrics, "inference_failed") > 0,
                        float(self.fixture.get("inference_timeout_s", 60.0))):
                    return self._finish(
                        False, "inference did not complete",
                        duplicate_or_regressed)
                if self._number(
                        self.metrics,
                        "last_completed_observation_id") < accepted:
                    return self._finish(
                        False, "production inference failed",
                        duplicate_or_regressed)
        self._spin_until(
            lambda: self._number(self.metrics, "worker_busy") == 0 and
            self._number(self.metrics, "backlog_size") == 0, 60.0)
        return self._finish(
            accepted == len(observations), "", duplicate_or_regressed)

    def _finish(self, passed, failure="", duplicate_or_regressed=0):
        observations = self.fixture.get("observations", [])
        metric = lambda key: self._number(self.metrics, key)
        float_metric = lambda key: self._float_number(self.metrics, key)
        completed = metric("inference_completed")
        stable_ids = {
            key for key, count in self.id_updates.items() if count > 1}
        stable_ids.update(
            value for value in self.metrics.get(
                "confirmed_track_ids", "").split(",") if value)
        stable_ids = sorted(stable_ids, key=int)
        acceptance = self.fixture.get("acceptance", {})
        reasons = [failure] if failure else []
        if metric("backlog_high_water_mark") > 1:
            reasons.append("latest-only backlog exceeded one")
        if metric("duplicate_or_regressed_submissions") != 0:
            reasons.append("Observation submission identity regressed")
        if metric("result_store_regressions") or metric(
                "duplicate_result_publish"):
            reasons.append("stale or duplicate completed result was published")
        if completed < int(acceptance.get("minimum_completed_results", 1)):
            reasons.append("too few completed results")
        if metric("tracks_lost") < int(
                acceptance.get("minimum_tracks_lost", 0)):
            reasons.append("expected lost lifecycle was not observed")
        lost_ids = sorted(
            {track_id for event in self.lost_id_events
             for track_id in event["lost_track_ids"]},
            key=int)
        expected_lost_ids = sorted(
            acceptance.get("expected_lost_track_ids", []), key=int)
        if expected_lost_ids and lost_ids != expected_lost_ids:
            reasons.append(
                "lost track identities did not match expected stable IDs")
        if len(stable_ids) < int(acceptance.get("minimum_stable_ids", 0)):
            reasons.append("expected stable track ID was not observed")
        summary = {
            "fixture_schema_version": self.fixture.get("schema_version"),
            "fixture_name": self.fixture.get("name", ""),
            "input_observation_count": len(observations),
            "accepted_observation_count": metric("latest_observation_id"),
            "first_source_stamp_ns": int(
                observations[0]["stamp_ns"]) if observations else None,
            "last_source_stamp_ns": int(
                observations[-1]["stamp_ns"]) if observations else None,
            "first_observation_id": (
                1 if metric("latest_observation_id") else None),
            "last_observation_id": metric("latest_observation_id"),
            "completed_result_count": completed,
            "object_lifecycle": {
                "appeared": metric("tracks_created"),
                "updated": metric("associations_matched"),
                "lost": metric("tracks_lost"),
                "stable_ids": stable_ids,
                "lost_track_ids": lost_ids,
                "lost_events": self.lost_id_events,
            },
            "stale_result_count": (
                metric("result_store_regressions") +
                metric("duplicate_result_publish")),
            "duplicate_or_regressed_source_timestamp_count": (
                duplicate_or_regressed),
            "geometry_quality": {
                "valid": metric("geometry_valid_total"),
                "degraded": metric("geometry_degraded_total"),
                "invalid": metric("geometry_invalid_total"),
                "invalid_intrinsics": metric("invalid_intrinsics_total"),
                "invalid_mask": metric("empty_mask_total"),
                "insufficient_depth": metric("insufficient_depth_total"),
                "empty_cloud": metric("empty_cloud_total"),
                "nonfinite": metric("nonfinite_geometry_total"),
            },
            "backpressure": {
                "backlog_high_water_mark": metric(
                    "backlog_high_water_mark"),
                "observations_skipped_before_inference": metric(
                    "observations_skipped_before_inference"),
            },
            "performance": {
                "execution_backend": os.getenv(
                    "EPD_EXECUTION_BACKEND", "legacy"),
                "inference_latency_min_ms": float_metric(
                    "inference_latency_min_ms"),
                "inference_latency_avg_ms": float_metric(
                    "inference_latency_avg_ms"),
                "inference_latency_max_ms": float_metric(
                    "inference_latency_max_ms"),
                "inference_rate_hz": float_metric("inference_rate_hz"),
                "observation_rate_hz": float_metric("observation_rate_hz"),
            },
            "result": "FAIL" if reasons or not passed else "PASS",
            "failures": [reason for reason in reasons if reason],
        }
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.summary_path.with_suffix(
            self.summary_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        os.replace(temporary, self.summary_path)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0 if summary["result"] == "PASS" else 1


def main():
    rclpy.init()
    node = None
    try:
        node = Replay()
        code = node.run()
    except Exception as error:
        print("EPD replay failed: " + str(error), file=sys.stderr, flush=True)
        code = 2
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
