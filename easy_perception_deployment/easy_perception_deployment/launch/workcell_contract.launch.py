from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _truthy(value):
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _launch_bridge(context):
    share = Path(get_package_share_directory("easy_perception_deployment"))
    script = share / "launch" / "workcell_contract_bridge.py"

    def value(name):
        return LaunchConfiguration(name).perform(context)

    command = [
        "python3",
        str(script),
        "--scene-id",
        value("scene_id"),
        "--camera-id",
        value("camera_id"),
        "--profile-ref",
        value("profile_ref"),
        "--runtime-mode",
        value("runtime_mode"),
        "--source-mode",
        value("source_mode"),
        "--localization-topic",
        value("localization_topic"),
        "--tracking-topic",
        value("tracking_topic"),
        "--diagnostics-topic",
        value("diagnostics_topic"),
        "--snapshot-topic",
        value("snapshot_topic"),
        "--status-topic",
        value("status_topic"),
        "--stale-timeout-s",
        value("stale_timeout_s"),
        "--status-period-s",
        value("status_period_s"),
    ]
    if _truthy(value("require_tracking_ids")):
        command.append("--require-tracking-ids")

    return [
        ExecuteProcess(
            cmd=command,
            output="screen",
            name="epd_workcell_contract_bridge",
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "scene_id",
            default_value="",
            description=(
                "Workcell Studio scene identity. Required for a valid normalized "
                "snapshot; EPD does not discover or own scene definitions."
            ),
        ),
        DeclareLaunchArgument(
            "camera_id",
            default_value="realsense_d435i_1",
            description="Workcell Studio camera identity for this EPD stream.",
        ),
        DeclareLaunchArgument(
            "profile_ref",
            default_value="",
            description=(
                "Optional EPD-5 perception-profile reference supplied by the scene."
            ),
        ),
        DeclareLaunchArgument(
            "runtime_mode",
            default_value="live",
            choices=["live", "replay"],
            description="Contract provenance mode.",
        ),
        DeclareLaunchArgument(
            "source_mode",
            default_value="tracking",
            choices=["tracking", "localization", "auto"],
            description=(
                "EPD P3 source. Tracking is recommended for Workcell Studio because "
                "it provides stable object IDs."
            ),
        ),
        DeclareLaunchArgument(
            "require_tracking_ids",
            default_value="true",
            description=(
                "Reject objects without stable Tracking IDs. Set false only for "
                "localization-only inspection/replay workflows."
            ),
        ),
        DeclareLaunchArgument(
            "localization_topic",
            default_value="/easy_perception_deployment/epd_localize_output",
        ),
        DeclareLaunchArgument(
            "tracking_topic",
            default_value="/easy_perception_deployment/epd_tracking_output",
        ),
        DeclareLaunchArgument(
            "diagnostics_topic",
            default_value="/easy_perception_deployment/inference_diagnostics",
        ),
        DeclareLaunchArgument(
            "snapshot_topic",
            default_value="/workcell_studio/epd_detection_snapshot_json",
        ),
        DeclareLaunchArgument(
            "status_topic",
            default_value="/workcell_studio/epd_connector_status",
        ),
        DeclareLaunchArgument("stale_timeout_s", default_value="2.0"),
        DeclareLaunchArgument("status_period_s", default_value="0.25"),
        OpaqueFunction(function=_launch_bridge),
    ])
