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

import importlib.util
from pathlib import Path
import sys

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
import pytest
import yaml


_LAUNCH_DIR = Path(__file__).resolve().parent
if str(_LAUNCH_DIR) not in sys.path:
    sys.path.insert(0, str(_LAUNCH_DIR))

from live_source_config import resolve_live_source  # noqa: E402


ALIGNED = {
    "rgb_input_topic": "/camera/camera/color/image_raw",
    "depth_input_topic": "/camera/camera/aligned_depth_to_color/image_raw",
    "camera_info_input_topic": "/camera/camera/color/camera_info",
    "depth_alignment": "camera",
}


def _ingress_parameters(launch_file, **arguments):
    spec = importlib.util.spec_from_file_location(launch_file, _LAUNCH_DIR / launch_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.get_package_share_directory = lambda _package: str(_LAUNCH_DIR.parent)
    description = module.generate_launch_description()
    context = LaunchContext()
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            default = "".join(part.text for part in entity.default_value)
            context.launch_configurations[entity.name] = arguments.get(entity.name, default)
    ingress = next(e for e in description.entities if isinstance(e, OpaqueFunction))
    (node,) = ingress.execute(context)
    (parameters,) = node._Node__parameters

    def text(value):
        # launch_ros stores each value as YAML-dumped TextSubstitutions.
        if isinstance(value, (list, tuple)):
            return yaml.safe_load("".join(part.text for part in value))
        return value

    return {text(key): text(value) for key, value in parameters.items()}


@pytest.mark.parametrize("launch_file", ["run.launch.py", "epd_emd_pipeline.launch.py"])
def test_default_source_keeps_the_driver_aligned_topics(launch_file):
    parameters = _ingress_parameters(launch_file)
    assert {k: parameters[k] for k in ALIGNED} == ALIGNED
    assert "depth_camera_info_input_topic" not in parameters


@pytest.mark.parametrize("launch_file", ["run.launch.py", "epd_emd_pipeline.launch.py"])
def test_unaligned_source_registers_raw_depth_with_the_depth_camera_info(launch_file):
    parameters = _ingress_parameters(launch_file, camera_source="realsense_unaligned")
    assert parameters["depth_input_topic"] == "/camera/camera/depth/image_rect_raw"
    assert parameters["depth_camera_info_input_topic"] == "/camera/camera/depth/camera_info"
    assert parameters["camera_info_input_topic"] == "/camera/camera/color/camera_info"
    assert parameters["depth_alignment"] == "register"
    assert parameters["max_rgbd_pair_skew_s"] == pytest.approx(0.020)
    # The EPD node still reads the one normalized ingress interface.
    assert parameters["depth_output_topic"] == (
        "/easy_perception_deployment/ingress/aligned_depth/image_raw")


def test_explicit_topics_override_the_preset():
    parameters = _ingress_parameters(
        "run.launch.py", camera_source="realsense_unaligned", rgb_topic="/cam2/color/image_raw")
    assert parameters["rgb_input_topic"] == "/cam2/color/image_raw"
    assert parameters["depth_alignment"] == "register"


def test_unknown_source_and_incomplete_register_configuration_are_refused():
    with pytest.raises(ValueError, match="Unknown camera_source"):
        resolve_live_source("webcam")
    with pytest.raises(ValueError, match="Unknown live source field"):
        resolve_live_source("realsense_aligned", {"frame": "x"})
    source = resolve_live_source("realsense_unaligned", {"depth_camera_info_topic": "  "})
    assert source["depth_camera_info_topic"] == "/camera/camera/depth/camera_info"
