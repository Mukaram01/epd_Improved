from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    rgb_topic = LaunchConfiguration("rgb_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    use_depth = LaunchConfiguration("use_depth")
    service_timeout_s = LaunchConfiguration("service_timeout_s")
    log_level = LaunchConfiguration("log_level")

    pkg_share = get_package_share_directory("easy_perception_deployment")
    ingress_rgb = "/easy_perception_deployment/ingress/color/image_raw"
    ingress_depth = "/easy_perception_deployment/ingress/aligned_depth/image_raw"
    ingress_info = "/easy_perception_deployment/ingress/color/camera_info"

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
            "use_depth",
            default_value="true",
            description=(
                "Set to 'false' to run without depth (e.g. when the aligned-depth "
                "topic has no publisher). 3D coordinates will be unavailable."
            )
        ),
        DeclareLaunchArgument(
            "service_timeout_s",
            default_value="10.0",
            description="Timeout covering fresh synchronized input plus one inference cycle"
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
            executable="epd_sensor_ingress",
            name="epd_sensor_ingress",
            output="screen",
            parameters=[{
                "rgb_input_topic": rgb_topic,
                "depth_input_topic": depth_topic,
                "camera_info_input_topic": camera_info_topic,
                "rgb_output_topic": ingress_rgb,
                "depth_output_topic": ingress_depth,
                "camera_info_output_topic": ingress_info,
            }],
        ),

        Node(
            package="easy_perception_deployment",
            executable="easy_perception_deployment",
            name="easy_perception_deployment",
            output="screen",
            emulate_tty=True,

            cwd=pkg_share,

            parameters=[
                {"use_depth": use_depth},
                {"rgb_topic": ingress_rgb},
                {"depth_topic": ingress_depth},
                {"camera_info_topic": ingress_info},
                {"service_timeout_s": service_timeout_s},
            ],
            remappings=[
                # Route the camera RGB topic into the node's primary image input.
                ("/easy_perception_deployment/image_input", ingress_rgb),
            ],
            arguments=["--ros-args", "--log-level", log_level],
        ),
    ])
