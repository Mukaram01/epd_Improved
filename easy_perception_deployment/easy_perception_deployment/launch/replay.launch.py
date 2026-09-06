from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _stop_at_eof(event, _context):
    if event.returncode != 0:
        raise RuntimeError(
            "EPD replay acceptance failed with exit code " + str(event.returncode))
    return [EmitEvent(event=Shutdown(reason="deterministic replay reached EOF"))]


def generate_launch_description():
    share = get_package_share_directory("easy_perception_deployment")
    fixture = LaunchConfiguration("fixture")
    mode = LaunchConfiguration("mode")
    summary = LaunchConfiguration("summary_output")
    publish_workcell_contract = LaunchConfiguration("publish_workcell_contract")
    workcell_scene_id = LaunchConfiguration("workcell_scene_id")
    workcell_camera_id = LaunchConfiguration("workcell_camera_id")
    perception_profile_ref = LaunchConfiguration("perception_profile_ref")

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
            "tracker_type_override": "MEDIANFLOW",
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
    workcell_contract = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            share + "/launch/workcell_contract.launch.py"),
        condition=IfCondition(publish_workcell_contract),
        launch_arguments={
            "scene_id": workcell_scene_id,
            "camera_id": workcell_camera_id,
            "profile_ref": perception_profile_ref,
            "runtime_mode": "replay",
            "source_mode": "tracking",
            "require_tracking_ids": "true",
        }.items(),
    )
    stop_at_eof = RegisterEventHandler(OnProcessExit(
        target_action=replay,
        on_exit=_stop_at_eof,
    ))
    return LaunchDescription([
        DeclareLaunchArgument(
            "fixture", default_value=share + "/fixtures/p8_tracking.json"),
        DeclareLaunchArgument("mode", default_value="fast"),
        DeclareLaunchArgument(
            "summary_output", default_value="/tmp/epd_replay_summary.json"),
        DeclareLaunchArgument(
            "publish_workcell_contract",
            default_value="false",
            description=(
                "Also publish workcell_perception_snapshot/v1 during replay."
            ),
        ),
        DeclareLaunchArgument(
            "workcell_scene_id",
            default_value="",
            description="Scene identity to stamp on normalized replay snapshots.",
        ),
        DeclareLaunchArgument(
            "workcell_camera_id",
            default_value="fixture_camera",
            description="Camera identity expected by the Workcell Studio scene.",
        ),
        DeclareLaunchArgument(
            "perception_profile_ref",
            default_value="",
            description="Optional EPD-5 profile reference for replay provenance.",
        ),
        production,
        replay,
        workcell_contract,
        stop_at_eof,
    ])
