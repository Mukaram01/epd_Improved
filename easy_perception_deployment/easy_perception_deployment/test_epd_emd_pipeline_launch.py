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

from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


LAUNCH_FILE = Path(__file__).parent / "launch" / "epd_emd_pipeline.launch.py"


def _description():
    spec = importlib.util.spec_from_file_location("epd_emd_pipeline_launch", LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.get_package_share_directory = lambda _package: str(Path(__file__).parent)
    return module.generate_launch_description()


def _default_text(argument):
    return "".join(part.text for part in argument.default_value)


def test_mode_override_defaults_preserve_persistent_configuration():
    arguments = {
        entity.name: entity
        for entity in _description().entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert _default_text(arguments["usecase_mode_override"]) == "-1"
    assert _default_text(arguments["tracker_type_override"]) == ""
    assert _default_text(arguments["tracking_maximum_missed_observations"]) == "30"
    # Camera topics now come from the camera_source preset; the default preset keeps the
    # driver-aligned topics (see launch/test_live_source_config.py).
    assert _default_text(arguments["camera_source"]) == "realsense_aligned"
    for topic in ("rgb_topic", "depth_topic", "camera_info_topic"):
        assert _default_text(arguments[topic]) == ""


def test_mode_overrides_are_typed_epd_node_parameters():
    epd_node = next(
        entity for entity in _description().entities
        if isinstance(entity, Node) and entity.node_executable == "easy_perception_deployment"
    )
    parameters = {
        "".join(part.text for part in key): value
        for parameter_set in epd_node._Node__parameters
        for key, value in parameter_set.items()
    }
    mode = parameters["usecase_mode_override"]
    tracker = parameters["tracker_type_override"]
    missed_observations = parameters["tracking_maximum_missed_observations"]
    assert isinstance(mode, ParameterValue) and mode.value_type is int
    assert isinstance(tracker, ParameterValue) and tracker.value_type is str
    assert isinstance(missed_observations, ParameterValue)
    assert missed_observations.value_type is int
    assert "".join(part.text for part in mode.value[0].variable_name) == (
        "usecase_mode_override")
    assert "".join(part.text for part in tracker.value[0].variable_name) == (
        "tracker_type_override")
    assert "".join(
        part.text for part in missed_observations.value[0].variable_name
    ) == "tracking_maximum_missed_observations"


def test_backend_follows_the_saved_deploy_selection_instead_of_the_legacy_gpu_path():
    entities = _description().entities
    arguments = {e.name: e for e in entities if isinstance(e, DeclareLaunchArgument)}
    # The package's saved session_config.json selects auto (CPU on a CPU-only build).
    assert _default_text(arguments["execution_backend"]) in ("auto", "cpu", "cuda", "tensorrt")
    assert _default_text(arguments["execution_backend_gpu_index"]).isdigit()
    exported = {
        "".join(part.text for part in e.name): e.value
        for e in entities if isinstance(e, SetEnvironmentVariable)
    }
    assert set(exported) == {"EPD_EXECUTION_BACKEND", "EPD_GPU_INDEX"}
