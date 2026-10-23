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

"""Regenerate the SYNTHETIC perception-session fixtures used by GUI tests.

The images are drawn shapes, not camera data, and every report is written
by hand in the exact key/value format of EPD::makeReadinessReport. They
exist to exercise readiness states and failure handling that the recorded
D435i set does not contain. They are not measurements.

Usage: python3 make_synthetic_sessions.py   (needs PySide6 only)
"""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage  # noqa
from PySide6.QtGui import QPainter  # noqa: E402

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 640, 360
FRAME_ID = "synthetic_color_optical_frame"
HARDWARE_ID = "synthetic_camera"
BASE_NS = 1_000_000_000_000  # 1000 s: obviously not a wall-clock stamp
PROVENANCE = {
    "kind": "synthetic",
    "summary": (
        "SYNTHETIC TEST DATA. Drawn shapes and hand-written readiness "
        "reports in the EPD::makeReadinessReport format. Not camera data "
        "and not a perception measurement."),
    "generator": "fixtures/perception_sessions/make_synthetic_sessions.py",
}

# name, shape, colour, roi(x, y, w, h)
OBJECTS = {
    "apple": ("ellipse", "#c0392b", (90, 170, 70, 70)),
    "banana": ("ellipse", "#e5c33b", (200, 180, 110, 50)),
    "bottle": ("rect", "#2e86c1", (360, 120, 50, 130)),
    "cup": ("rect", "#a569bd", (460, 190, 70, 70)),
    "banana_overlap": ("ellipse", "#d4ac0d", (250, 190, 100, 45)),
}


def _status(index, state, cls, conf, roi, mask, depth, centroid="",
            axis="", reasons="none", note="", track="", obs=1):
    level = {"GRASP_CANDIDATE": 0, "POSE_VALID": 0, "REJECTED": 2}.get(
        state, 1)
    positioned = state in ("LOCALISED", "POSE_VALID", "GRASP_CANDIDATE")
    oriented = state in ("POSE_VALID", "GRASP_CANDIDATE")
    values = [
        ("state", state), ("class", cls), ("confidence", conf),
        ("roi", ",".join(str(v) for v in roi)),
        ("mask_pixels", str(mask)), ("valid_depth_pixels", str(depth)),
        ("position_valid", "true" if positioned else "false"),
        ("orientation_valid", "true" if oriented else "false"),
        ("centroid_m", centroid), ("axis", axis), ("reasons", reasons),
        ("note", note), ("track_id", track), ("observation_id", str(obs)),
    ]
    return {
        "level": level,
        "name": f"easy_perception_deployment/detection/{index}",
        "message": state,
        "hardware_id": HARDWARE_ID,
        "values": [{"key": k, "value": v} for k, v in values],
    }


def _report(stamp_ns, statuses):
    return {
        "header": {
            "stamp": {"sec": stamp_ns // 10**9, "nanosec": stamp_ns % 10**9},
            "frame_id": FRAME_ID,
        },
        "status": statuses,
    }


def _shift(roi, dx):
    return (roi[0] + dx, roi[1], roi[2], roi[3])


def _all_states(obs, dx):
    o = {k: _shift(v[2], dx) for k, v in OBJECTS.items()}
    return [
        _status(0, "GRASP_CANDIDATE", "apple", "0.954210", o["apple"],
                3810, 3702, "0.081200,-0.032100,0.512300",
                "0.120000,0.990000,0.070000", track="1", obs=obs),
        _status(1, "LOCALISED", "banana", "0.883104", o["banana"], 4310,
                4122, "-0.010400,0.021700,0.498800", "",
                "degenerate_axis,ambiguous_axis_no_pose", track="2",
                obs=obs),
        _status(2, "POSE_VALID", "bottle", "0.702551", o["bottle"], 5120,
                5004, "0.090300,0.180600,0.522100",
                "0.010000,0.999000,0.040000", note="not_selected",
                track="3", obs=obs),
        _status(3, "DETECTED", "cup", "0.611872", o["cup"], 3900, 41, "",
                "", "insufficient_depth", track="4", obs=obs),
        _status(4, "REJECTED", "banana", "0.574390", o["banana_overlap"],
                3320, 3100, "", "", "overlapping_ownership", track="5",
                obs=obs),
    ]


def _draw(path, dx, caption, objects=True):
    image = QImage(WIDTH, HEIGHT, QImage.Format_RGB888)
    image.fill(QColor("#2b2f36"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.fillRect(QRect(30, 110, 580, 200), QColor("#5d4e3f"))
    if objects:
        for shape, colour, roi in OBJECTS.values():
            x, y, w, h = _shift(roi, dx)
            painter.setBrush(QColor(colour))
            painter.setPen(Qt.NoPen)
            if shape == "ellipse":
                painter.drawEllipse(QRect(x, y, w, h))
            else:
                painter.drawRoundedRect(QRect(x, y, w, h), 8, 8)
    painter.setPen(QColor("#ffffff"))
    font = QFont()
    font.setPixelSize(18)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(QRect(0, 14, WIDTH, 26), Qt.AlignCenter,
                     "SYNTHETIC TEST DATA - not a camera image")
    font.setPixelSize(14)
    font.setBold(False)
    painter.setFont(font)
    painter.drawText(QRect(0, 44, WIDTH, 22), Qt.AlignCenter, caption)
    painter.end()
    if not image.save(str(path), "PNG"):
        raise RuntimeError(f"cannot write {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(directory, name, stamp_ns, dx, caption, objects=True):
    frames = directory / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    path = frames / name
    digest = _draw(path, dx, caption, objects)
    return {
        "path": f"frames/{name}",
        "sha256": digest,
        "header": {
            "stamp": {"sec": stamp_ns // 10**9, "nanosec": stamp_ns % 10**9},
            "frame_id": FRAME_ID,
        },
        "width": WIDTH,
        "height": HEIGHT,
        "encoding": "rgb8",
    }


def _write(directory, name, frames):
    manifest = {
        "schema": "epd.perception_session",
        "schema_version": 1,
        "name": name,
        "provenance": PROVENANCE,
        "readiness_topic": "/easy_perception_deployment/detection_readiness",
        "image_topic": "/synthetic/color/image_raw",
        "frames": frames,
    }
    (directory / "session.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def all_states():
    directory = HERE / "synthetic_readiness_states"
    step = 500_000_000
    t = [BASE_NS + i * step for i in range(6)]
    frames = [
        {"label": "All five readiness states",
         "image": _image(directory, "000.png", t[0], 0,
                         "Frame 0: five proposals, five readiness states"),
         "readiness": _report(t[0], _all_states(1, 0))},
        {"label": "Same tracks, objects moved",
         "image": _image(directory, "001.png", t[1], 12,
                         "Frame 1: same track IDs after a small move"),
         "readiness": _report(t[1], _all_states(2, 12))},
        {"label": "Readiness report missing",
         "image": _image(directory, "002.png", t[2], 12,
                         "Frame 2: no readiness report was published"),
         "readiness": None},
        {"label": "Deliberately mismatched pairing",
         "image": _image(directory, "003.png", t[3], 24,
                         "Frame 3: report belongs to an older image"),
         "readiness": _report(t[2], _all_states(3, 12))},
        {"label": "RGB frame missing",
         "image": None,
         "readiness": _report(t[4], _all_states(4, 24))},
        {"label": "Report with zero detections",
         "image": _image(directory, "005.png", t[5], 0,
                         "Frame 5: empty table, report has 0 proposals",
                         objects=False),
         "readiness": _report(t[5], [])},
    ]
    _write(directory, "Synthetic readiness states (test fixture)", frames)


def no_readiness():
    directory = HERE / "synthetic_no_readiness"
    t = [BASE_NS + i * 500_000_000 for i in range(2)]
    frames = [
        {"label": "Readiness topic not recorded",
         "image": _image(directory, f"{i:03d}.png", stamp, 0,
                         "Recording without the readiness topic"),
         "readiness": None}
        for i, stamp in enumerate(t)
    ]
    _write(directory, "Synthetic recording without readiness (test fixture)",
           frames)


if __name__ == "__main__":
    app = QGuiApplication([])
    all_states()
    no_readiness()
    print("wrote", HERE / "synthetic_readiness_states", "and",
          HERE / "synthetic_no_readiness")
