from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rgb_topic = LaunchConfiguration("rgb_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    image_transport = LaunchConfiguration("image_transport")
    log_level = LaunchConfiguration("log_level")

    return LaunchDescription([
        DeclareLaunchArgument(
            "rgb_topic",
            default_value="/camera/camera/color/image_raw",
            description="RGB image topic"
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/camera/color/camera_info",
            description="Camera info topic"
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/camera/camera/aligned_depth_to_color/image_raw",
            description="Aligned depth topic"
        ),
        DeclareLaunchArgument(
            "image_transport",
            default_value="raw",
            description="Image transport plugin (raw/compressed)"
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="info",
            description="debug/info/warn/error/fatal"
        ),

        Node(
            package="easy_perception_deployment",
            executable="easy_perception_deployment",
            name="easy_perception_deployment",
            output="screen",
            emulate_tty=True,
            remappings=[
                ("/camera/color/image_raw", rgb_topic),
                ("/camera/color/camera_info", camera_info_topic),

                # IMPORTANT: this is what your node actually subscribes to (from ros2 node info)
                ("/camera/depth/image_rect_raw", depth_topic),

                # Keep for safety if code also uses this name internally
                ("/camera/aligned_depth_to_color/image_raw", depth_topic),
            ],
            parameters=[{
                "image_transport": image_transport,
            }],
            arguments=["--ros-args", "--log-level", log_level],
        ),
    ])
