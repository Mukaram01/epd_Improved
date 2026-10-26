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

from pathlib import Path
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


_LAUNCH_DIR = Path(__file__).resolve().parent
if str(_LAUNCH_DIR) not in sys.path:
    sys.path.insert(0, str(_LAUNCH_DIR))

from backend_launch_config import resolve_backend_launch_defaults  # noqa: E402
from live_source_config import (  # noqa: E402
    DEFAULT_LIVE_SOURCE, LIVE_SOURCES, ingress_parameters, resolve_live_source)


def _sensor_ingress(context, rgb_out, depth_out, info_out):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    source = resolve_live_source(arg("camera_source"), {
        "rgb_topic": arg("rgb_topic"),
        "camera_info_topic": arg("camera_info_topic"),
        "depth_topic": arg("depth_topic"),
        "depth_camera_info_topic": arg("depth_camera_info_topic"),
    })
    parameters = ingress_parameters(source, rgb_out, depth_out, info_out)
    parameters["max_rgbd_pair_skew_s"] = float(arg("max_rgbd_pair_skew_s"))
    return [Node(
        package="easy_perception_deployment",
        executable="epd_sensor_ingress",
        name="epd_sensor_ingress",
        output="screen",
        parameters=[parameters],
    )]


def generate_launch_description():
    use_depth = LaunchConfiguration("use_depth")
    image_output_qos_reliability = LaunchConfiguration(
        "image_output_qos_reliability"
    )
    slow_frame_warn_ms = LaunchConfiguration("slow_frame_warn_ms")
    max_processing_fps = LaunchConfiguration("max_processing_fps")
    service_timeout_s = LaunchConfiguration("service_timeout_s")
    log_level = LaunchConfiguration("log_level")
    execution_backend = LaunchConfiguration("execution_backend")
    execution_backend_gpu_index = LaunchConfiguration(
        "execution_backend_gpu_index"
    )

    # Make the node run with its working directory set to the package share
    # directory. Then relative paths like ./data/model/... work reliably.
    pkg_share = get_package_share_directory("easy_perception_deployment")
    session_config = Path(pkg_share) / "config" / "session_config.json"
    backend_default, gpu_index_default = resolve_backend_launch_defaults(
        session_config
    )

    ingress_rgb = "/easy_perception_deployment/ingress/color/image_raw"
    ingress_depth = "/easy_perception_deployment/ingress/aligned_depth/image_raw"
    ingress_info = "/easy_perception_deployment/ingress/color/camera_info"

    return LaunchDescription([
        DeclareLaunchArgument(
            "camera_source",
            default_value=DEFAULT_LIVE_SOURCE,
            choices=sorted(LIVE_SOURCES),
            description=(
                "Live camera source: realsense_aligned forwards the driver's "
                "aligned depth; realsense_unaligned registers depth/image_rect_raw "
                "into the colour frame inside the EPD ingress."
            )
        ),
        DeclareLaunchArgument(
            "depth_camera_info_topic",
            default_value="",
            description="Depth camera info topic (register mode); empty uses the preset"
        ),
        DeclareLaunchArgument(
            "max_rgbd_pair_skew_s",
            default_value="0.020",
            description="Register mode: maximum colour/depth sensor-stamp skew to pair"
        ),
        DeclareLaunchArgument(
            "rgb_topic",
            default_value="",
            description="RGB image topic; empty uses the camera_source preset"
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="",
            description="Colour camera info topic; empty uses the camera_source preset"
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="",
            description=(
                "Depth topic (aligned, or raw when the source registers depth); "
                "empty uses the camera_source preset"
            )
        ),
        DeclareLaunchArgument(
            "use_depth",
            default_value="true",
            description=(
                "Set to 'false' to run without depth (e.g. when the aligned-depth "
                "topic has no publisher). 3D coordinates will be unavailable."
            )
        ),
        DeclareLaunchArgument(
            "image_output_qos_reliability",
            default_value="best_effort",
            description=(
                "QoS reliability for /easy_perception_deployment/image_output "
                "('best_effort' for low latency, 'reliable' for debug viewers)."
            )
        ),
        DeclareLaunchArgument(
            "slow_frame_warn_ms",
            default_value="1000",
            description=(
                "Warn (throttled) when single-frame inference latency exceeds "
                "this many ms. Set <=0 to disable."
            )
        ),
        DeclareLaunchArgument(
            "max_processing_fps",
            default_value="0.0",
            description=(
                "Optional cap on processed FPS. Frames arriving sooner than 1/fps "
                "after the last processed frame are dropped. Set 0 to disable."
            )
        ),
        DeclareLaunchArgument(
            "service_timeout_s",
            default_value="10.0",
            description=(
                "Timeout covering fresh synchronized input plus one inference cycle"
            )
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="info",
            description="debug/info/warn/error/fatal"
        ),
        DeclareLaunchArgument(
            "execution_backend",
            default_value=backend_default,
            description=(
                "ONNX Runtime backend: auto, cpu, cuda, or tensorrt. Defaults "
                "to the Deploy selection saved in session_config.json."
            )
        ),
        DeclareLaunchArgument(
            "execution_backend_gpu_index",
            default_value=gpu_index_default,
            description=(
                "GPU index for CUDA/TensorRT. Defaults to the Deploy selection."
            )
        ),

        # OrtBase reads these variables. Setting them here closes the gap where
        # a plain `ros2 launch` ignored the backend selected in the Deploy GUI
        # and fell back to the legacy gpuIdx path.
        SetEnvironmentVariable(
            name="EPD_EXECUTION_BACKEND",
            value=execution_backend,
        ),
        SetEnvironmentVariable(
            name="EPD_GPU_INDEX",
            value=execution_backend_gpu_index,
        ),

        OpaqueFunction(
            function=_sensor_ingress,
            args=[ingress_rgb, ingress_depth, ingress_info],
        ),

        Node(
            package="easy_perception_deployment",
            executable="easy_perception_deployment",
            name="easy_perception_deployment",
            output="screen",
            emulate_tty=True,

            # Critical: the runtime config/model paths are package-relative.
            cwd=pkg_share,

            parameters=[
                {"use_depth": use_depth},
                {"rgb_topic": ingress_rgb},
                {"depth_topic": ingress_depth},
                {"camera_info_topic": ingress_info},
                {"image_output_qos_reliability": image_output_qos_reliability},
                {"slow_frame_warn_ms": slow_frame_warn_ms},
                {"max_processing_fps": max_processing_fps},
                {"service_timeout_s": service_timeout_s},
            ],
            remappings=[
                # Route the camera RGB topic into the node's primary image input.
                ("/easy_perception_deployment/image_input", ingress_rgb),
            ],
            arguments=["--ros-args", "--log-level", log_level],
        ),
    ])
