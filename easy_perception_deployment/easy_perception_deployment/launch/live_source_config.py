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

"""Named live camera sources for the EPD sensor ingress.

A source names the camera topics and how depth reaches the colour frame:

``realsense_aligned``
    The RealSense driver aligns depth (``align_depth.enable:=true``); the ingress
    forwards ``aligned_depth_to_color/image_raw`` untouched. This is the default
    and the behaviour of every earlier release.
``realsense_unaligned``
    The driver publishes ``depth/image_rect_raw`` in the depth optical frame; the
    ingress registers it into the colour optical frame using both CameraInfos and
    the driver's static depth->colour TF.

Explicit topic launch arguments override the selected source's topics.
"""

from __future__ import annotations


LIVE_SOURCES = {
    "realsense_aligned": {
        "rgb_topic": "/camera/camera/color/image_raw",
        "camera_info_topic": "/camera/camera/color/camera_info",
        "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
        "depth_camera_info_topic": "",
        "depth_alignment": "camera",
    },
    "realsense_unaligned": {
        "rgb_topic": "/camera/camera/color/image_raw",
        "camera_info_topic": "/camera/camera/color/camera_info",
        "depth_topic": "/camera/camera/depth/image_rect_raw",
        "depth_camera_info_topic": "/camera/camera/depth/camera_info",
        "depth_alignment": "register",
    },
}
DEFAULT_LIVE_SOURCE = "realsense_aligned"


def resolve_live_source(name, overrides=None):
    """Return the ingress topics for ``name`` with non-empty ``overrides`` applied."""
    key = str(name or "").strip()
    if key not in LIVE_SOURCES:
        raise ValueError(
            f"Unknown camera_source '{key}'. Choose one of: {', '.join(sorted(LIVE_SOURCES))}")
    resolved = dict(LIVE_SOURCES[key])
    for field, value in (overrides or {}).items():
        if field not in resolved:
            raise ValueError(f"Unknown live source field '{field}'")
        if str(value).strip():
            resolved[field] = str(value).strip()
    if resolved["depth_alignment"] == "register" and not resolved["depth_camera_info_topic"]:
        raise ValueError("depth_alignment 'register' needs depth_camera_info_topic")
    return resolved


def ingress_parameters(source, rgb_out, depth_out, info_out):
    """Map a resolved source onto epd_sensor_ingress parameters."""
    parameters = {
        "rgb_input_topic": source["rgb_topic"],
        "depth_input_topic": source["depth_topic"],
        "camera_info_input_topic": source["camera_info_topic"],
        "rgb_output_topic": rgb_out,
        "depth_output_topic": depth_out,
        "camera_info_output_topic": info_out,
        "depth_alignment": source["depth_alignment"],
    }
    if source["depth_alignment"] == "register":
        parameters["depth_camera_info_input_topic"] = source["depth_camera_info_topic"]
    return parameters
