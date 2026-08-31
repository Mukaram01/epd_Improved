#!/usr/bin/env python3
"""Bounded P0/P1 runtime acceptance probe for EPD camera ingress."""

import argparse
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from epd_msgs.msg import EPDObjectLocalization, EPDObjectTracking
from diagnostic_msgs.msg import DiagnosticArray


def stamp_ns(message):
    stamp = message.header.stamp
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class StreamState:
    def __init__(self):
        self.count = 0
        self.first_stamp_ns = None
        self.last_stamp_ns = None
        self.last_receive_monotonic = None
        self.duplicate_stamps = 0
        self.regressed_stamps = 0

    def observe(self, message):
        value = stamp_ns(message)
        if self.last_stamp_ns is not None:
            self.duplicate_stamps += int(value == self.last_stamp_ns)
            self.regressed_stamps += int(value < self.last_stamp_ns)
        self.count += 1
        self.first_stamp_ns = value if self.first_stamp_ns is None else self.first_stamp_ns
        self.last_stamp_ns = value
        self.last_receive_monotonic = time.monotonic()

    def report(self, now):
        return {
            "count": self.count,
            "first_stamp_ns": self.first_stamp_ns,
            "last_stamp_ns": self.last_stamp_ns,
            "age_s": None if self.last_receive_monotonic is None else now - self.last_receive_monotonic,
            "duplicate_stamps": self.duplicate_stamps,
            "regressed_stamps": self.regressed_stamps,
        }


class SoakProbe(Node):
    def __init__(self, pids):
        super().__init__("epd_soak_probe")
        self.started = time.monotonic()
        self.pids = pids
        self.start_rss_kib = {str(pid): None for pid in pids}
        self.max_rss_kib = {str(pid): 0 for pid in pids}
        self.end_rss_kib = {str(pid): None for pid in pids}
        self.inference_metrics_start = None
        self.inference_metrics_end = None
        self.streams = {
            "raw_rgb": StreamState(), "raw_depth": StreamState(), "raw_info": StreamState(),
            "ingress_rgb": StreamState(), "ingress_depth": StreamState(),
            "ingress_info": StreamState(), "localization_output": StreamState(),
            "tracking_output": StreamState(), "image_output": StreamState(),
        }
        self.stamp_sets = {
            "raw_rgb": set(), "raw_depth": set(), "raw_info": set(),
            "ingress_rgb": set(), "ingress_depth": set(), "ingress_info": set(),
        }
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        subscriptions = [
            (Image, "/camera/camera/color/image_raw", "raw_rgb"),
            (Image, "/camera/camera/aligned_depth_to_color/image_raw", "raw_depth"),
            (CameraInfo, "/camera/camera/color/camera_info", "raw_info"),
            (Image, "/easy_perception_deployment/ingress/color/image_raw", "ingress_rgb"),
            (Image, "/easy_perception_deployment/ingress/aligned_depth/image_raw", "ingress_depth"),
            (CameraInfo, "/easy_perception_deployment/ingress/color/camera_info", "ingress_info"),
            (EPDObjectLocalization, "/easy_perception_deployment/epd_localize_output",
             "localization_output"),
            (EPDObjectTracking, "/easy_perception_deployment/epd_tracking_output",
             "tracking_output"),
            (Image, "/easy_perception_deployment/image_output", "image_output"),
        ]
        self.subscriptions_ = [
            self.create_subscription(msg_type, topic, self.callback(name), qos)
            for msg_type, topic, name in subscriptions
        ]
        self.subscriptions_.append(self.create_subscription(
            DiagnosticArray,
            "/easy_perception_deployment/inference_diagnostics",
            self.inference_diagnostics_callback,
            QoSProfile(depth=1)))
        self.sample_timer = self.create_timer(1.0, self.sample_rss)

    def inference_diagnostics_callback(self, message):
        for status in message.status:
            if status.name != "easy_perception_deployment/inference_worker":
                continue
            metrics = {item.key: item.value for item in status.values}
            if self.inference_metrics_start is None:
                self.inference_metrics_start = metrics
            self.inference_metrics_end = metrics

    def callback(self, name):
        def receive(message):
            self.streams[name].observe(message)
            if name in self.stamp_sets:
                stamps = self.stamp_sets[name]
                stamps.add(stamp_ns(message))
                if len(stamps) > 4096:
                    cutoff = sorted(stamps)[-2048]
                    self.stamp_sets[name] = {item for item in stamps if item >= cutoff}
        return receive

    def sample_rss(self):
        for pid in self.pids:
            try:
                with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
                    for line in handle:
                        if line.startswith("VmRSS:"):
                            rss = int(line.split()[1])
                            key = str(pid)
                            if self.start_rss_kib[key] is None:
                                self.start_rss_kib[key] = rss
                            self.max_rss_kib[key] = max(self.max_rss_kib[key], rss)
                            self.end_rss_kib[key] = rss
                            break
            except (FileNotFoundError, ProcessLookupError):
                pass

    def report(self):
        now = time.monotonic()
        exact = set.intersection(
            self.stamp_sets["raw_rgb"], self.stamp_sets["raw_depth"],
            self.stamp_sets["raw_info"])
        relay_matches = {
            stream: len(self.stamp_sets[f"raw_{stream}"] & self.stamp_sets[f"ingress_{stream}"])
            for stream in ("rgb", "depth", "info")
        }
        return {
            "duration_s": now - self.started,
            "streams": {name: state.report(now) for name, state in self.streams.items()},
            "raw_exact_triplets_retained": len(exact),
            "source_to_ingress_stamp_matches": relay_matches,
            "start_rss_kib": self.start_rss_kib,
            "max_rss_kib": self.max_rss_kib,
            "end_rss_kib": self.end_rss_kib,
            "inference_metrics_start": self.inference_metrics_start,
            "inference_metrics_end": self.inference_metrics_end,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--output", default="epd_soak_result.json")
    parser.add_argument("--pid", action="append", type=int, default=[])
    args = parser.parse_args()
    rclpy.init()
    node = SoakProbe(args.pid)
    deadline = time.monotonic() + args.duration_s
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=1.0)
    finally:
        result = node.report()
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
