from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    rgb_topic = LaunchConfiguration("rgb_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    log_level = LaunchConfiguration("log_level")

    pkg_share = get_package_share_directory("easy_perception_deployment")

    return LaunchDescription([
        DeclareLaunchArgument(
            "rgb_topic",
            default_value="/camera/camera/color/image_raw",
            description="RGB image topic (RealSense D435i default)"
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/camera/color/camera_info",
            description="Camera info topic (RealSense D435i default)"
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/camera/camera/aligned_depth_to_color/image_raw",
            description="Aligned depth topic (RealSense D435i default)"
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="info",
            description="debug/info/warn/error/fatal"
        ),

        LogInfo(
            msg=(
                "[epd_emd_pipeline] EPD is starting in LOCALISATION_MODE (usecase_mode=3). "
                "Configure Easy Manipulator Improved (EMD) with the following topics: "
                "epd_localization_topic: /easy_perception_deployment/epd_localize_output  "
                "epd_tracking_topic: /easy_perception_deployment/epd_tracking_output"
            )
        ),

        Node(
            package="easy_perception_deployment",
            executable="easy_perception_deployment",
            name="easy_perception_deployment",
            output="screen",
            emulate_tty=True,

            cwd=pkg_share,

            remappings=[
                ("/camera/color/image_raw", rgb_topic),
                ("/camera/color/camera_info", camera_info_topic),
                # Both depth topic names are remapped so the node works whether
                # the internal code subscribes to the rect_raw or aligned name.
                ("/camera/depth/image_rect_raw", depth_topic),
                ("/camera/aligned_depth_to_color/image_raw", depth_topic),
            ],
            arguments=["--ros-args", "--log-level", log_level],
        ),
    ])
