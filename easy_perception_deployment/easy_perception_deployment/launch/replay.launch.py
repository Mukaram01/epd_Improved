from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _stop_at_eof(event, _context):
    if event.returncode != 0:
        raise RuntimeError("EPD replay acceptance failed with exit code " + str(event.returncode))
    return [EmitEvent(event=Shutdown(reason="deterministic replay reached EOF"))]


def generate_launch_description():
    share = get_package_share_directory("easy_perception_deployment")
    fixture = LaunchConfiguration("fixture")
    mode = LaunchConfiguration("mode")
    summary = LaunchConfiguration("summary_output")

    production = Node(
        package="easy_perception_deployment",
        executable="easy_perception_deployment",
        name="easy_perception_deployment",
        output="screen",
        cwd=share,
        parameters=[{
            "use_depth": True,
            "camera_id": "fixture_camera",
            "rgb_topic": "/easy_perception_deployment/ingress/color/image_raw",
            "depth_topic": "/easy_perception_deployment/ingress/aligned_depth/image_raw",
            "camera_info_topic": "/easy_perception_deployment/ingress/color/camera_info",
            "rgb_input_watchdog_timeout_s": 0.0,
            "usecase_mode_override": 4,
        }],
        remappings=[
            ("/easy_perception_deployment/image_input",
             "/easy_perception_deployment/ingress/color/image_raw")],
    )
    replay = Node(
        package="easy_perception_deployment",
        executable="epd_replay.py",
        name="epd_replay",
        output="screen",
        parameters=[{"fixture": fixture, "mode": mode, "summary_output": summary}],
    )
    stop_at_eof = RegisterEventHandler(OnProcessExit(
        target_action=replay,
        on_exit=_stop_at_eof,
    ))
    return LaunchDescription([
        DeclareLaunchArgument(
            "fixture", default_value=share + "/fixtures/p8_tracking.json"),
        DeclareLaunchArgument("mode", default_value="fast"),
        DeclareLaunchArgument("summary_output", default_value="/tmp/epd_replay_summary.json"),
        production, replay, stop_at_eof,
    ])
