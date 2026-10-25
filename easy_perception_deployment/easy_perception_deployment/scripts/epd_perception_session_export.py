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

"""Export a rosbag2 recording to a perception session for Deploy replay.

Reads /easy_perception_deployment/detection_readiness and one RGB topic,
pairs each report with the RGB frame that has the identical source header,
and writes ``session.json`` plus PNG frames. Readiness messages are stored
losslessly; nothing is re-computed. A report without an exactly matching
RGB frame is kept and marked as missing its image.

Example:
  ros2 bag record -o run1 /camera/camera/color/image_raw \\
      /easy_perception_deployment/detection_readiness
  epd_perception_session_export.py --bag run1 --output run1_session \\
      --image-topic /camera/camera/color/image_raw
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
import yaml
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Image

READINESS_TOPIC = "/easy_perception_deployment/detection_readiness"
IMAGE_TYPES = {
    "sensor_msgs/msg/Image": Image,
    "sensor_msgs/msg/CompressedImage": CompressedImage,
}


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def frame_stamp_ns(frame):
    stamp = (frame["image"] or frame["readiness"])["header"]["stamp"]
    return int(stamp["sec"]) * 1_000_000_000 + int(stamp["nanosec"])


def readiness_dict(message):
    header = message.header
    return {
        "header": {
            "stamp": {"sec": int(header.stamp.sec),
                      "nanosec": int(header.stamp.nanosec)},
            "frame_id": str(header.frame_id),
        },
        "status": [
            {
                "level": (status.level[0]
                          if isinstance(status.level, (bytes, bytearray))
                          else int(status.level)),
                "name": status.name,
                "message": status.message,
                "hardware_id": status.hardware_id,
                "values": [{"key": item.key, "value": item.value}
                           for item in status.values],
            }
            for status in message.status
        ],
    }


def open_reader(bag, topics):
    metadata = yaml.safe_load(
        (Path(bag) / "metadata.yaml").read_text(encoding="utf-8"))
    storage = metadata["rosbag2_bagfile_information"]["storage_identifier"]
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    return reader, types


def image_array(message):
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    encoding = message.encoding.lower()
    if encoding not in channels:
        raise ValueError(f"unsupported RGB encoding {message.encoding}")
    count = channels[encoding]
    rows = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(
        message.height, message.step)
    pixels = rows[:, :message.width * count].reshape(
        message.height, message.width, count)
    conversions = {
        "rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR,
        "bgra8": cv2.COLOR_BGRA2BGR,
    }
    if encoding in conversions:
        return cv2.cvtColor(pixels, conversions[encoding])
    if encoding == "mono8":
        return pixels[:, :, 0]
    return pixels


def write_image(message, directory, position):
    if isinstance(message, CompressedImage):
        suffix = ".png" if "png" in message.format.lower() else ".jpg"
        path = directory / f"{position:05d}{suffix}"
        path.write_bytes(bytes(message.data))
        decoded = cv2.imdecode(
            np.frombuffer(bytes(message.data), np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("compressed RGB frame cannot be decoded")
        height, width = decoded.shape[:2]
        encoding = message.format
    else:
        path = directory / f"{position:05d}.png"
        if not cv2.imwrite(str(path), image_array(message)):
            raise ValueError(f"cannot write {path}")
        width, height, encoding = message.width, message.height, (
            message.encoding)
    return {
        "path": f"frames/{path.name}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "header": {
            "stamp": {"sec": int(message.header.stamp.sec),
                      "nanosec": int(message.header.stamp.nanosec)},
            "frame_id": str(message.header.frame_id),
        },
        "width": int(width),
        "height": int(height),
        "encoding": str(encoding),
    }


def export(bag, output, image_topic, name, note, max_frames):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    reader, types = open_reader(bag, [READINESS_TOPIC])
    reports = []
    while reader.has_next():
        _topic, data, _time = reader.read_next()
        message = deserialize_message(data, DiagnosticArray)
        reports.append(
            ((stamp_ns(message.header), message.header.frame_id),
             readiness_dict(message)))
    del reader
    reports = reports[:max_frames]
    wanted = {key for key, _report in reports}

    reader, types = open_reader(bag, [image_topic])
    if image_topic not in types:
        raise SystemExit(f"{image_topic} is not in {bag}")
    message_type = IMAGE_TYPES.get(types[image_topic])
    if message_type is None:
        raise SystemExit(f"{image_topic} has unsupported type "
                         f"{types[image_topic]}")
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    images = {}
    unreported = []
    while reader.has_next():
        _topic, data, _time = reader.read_next()
        message = deserialize_message(data, message_type)
        key = (stamp_ns(message.header), message.header.frame_id)
        if key in images:
            continue
        if key in wanted:
            images[key] = write_image(message, frames_dir, len(images))
        elif not reports and len(unreported) < max_frames:
            unreported.append(
                (key[0], write_image(message, frames_dir,
                                     len(unreported))))
    del reader

    frames = []
    for key, report in reports:
        image = images.get(key)
        frames.append({
            "label": "" if image else "RGB frame missing",
            "image": image,
            "readiness": report,
        })
    for _stamp, image in sorted(unreported, key=lambda item: item[0]):
        frames.append({"label": "Readiness topic not recorded",
                       "image": image, "readiness": None})
    if not frames:
        raise SystemExit("no readiness reports or RGB frames to export")
    frames.sort(key=frame_stamp_ns)
    metadata = Path(bag) / "metadata.yaml"
    manifest = {
        "schema": "epd.perception_session",
        "schema_version": 1,
        "name": name or Path(bag).name,
        "provenance": {
            "kind": "recorded",
            "summary": note or "Exported from a rosbag2 recording.",
            "bag": str(Path(bag).resolve()),
            "bag_metadata_sha256": hashlib.sha256(
                metadata.read_bytes()).hexdigest(),
            "exporter": "epd_perception_session_export.py",
            "readiness_reports": len(reports),
            "reports_without_rgb": sum(
                1 for frame in frames
                if frame["readiness"] is not None and not frame["image"]),
        },
        "readiness_topic": READINESS_TOPIC,
        "image_topic": image_topic,
        "frames": frames,
    }
    (output / "session.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-topic", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--max-frames", type=int, default=600)
    args = parser.parse_args(argv)
    manifest = export(args.bag, args.output, args.image_topic, args.name,
                      args.note, args.max_frames)
    provenance = manifest["provenance"]
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "frames": len(manifest["frames"]),
        "readiness_reports": provenance["readiness_reports"],
        "reports_without_rgb": provenance["reports_without_rgb"],
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
