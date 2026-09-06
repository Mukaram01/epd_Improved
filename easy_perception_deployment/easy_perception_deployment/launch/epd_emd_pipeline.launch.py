from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    rgb_topic = LaunchConfiguration("rgb_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    use_depth = LaunchConfiguration("use_depth")
    service_timeout_s = LaunchConfiguration("service_timeout_s")
    log_level = LaunchConfiguration("log_level")
    usecase_mode_override = LaunchConfiguration("usecase_mode_override")
    tracker_type_override = LaunchConfiguration("tracker_type_override")
    tracking_maximum_missed_observations = LaunchConfiguration(
        "tracking_maximum_missed_observations")
    publish_workcell_contract = LaunchConfiguration("publish_workcell_contract")
    workcell_scene_id = LaunchConfiguration("workcell_scene_id")
    workcell_camera_id = LaunchConfiguration("workcell_camera_id")
    perception_profile_ref = LaunchConfiguration("perception_profile_ref")
    contract_source_mode = LaunchConfiguration("contract_source_mode")
    contract_require_tracking_ids = LaunchConfiguration(
        "contract_require_tracking_ids")

    pkg_share = get_package_share_directory("easy_perception_deployment")
    ingress_rgb = "/easy_perception_deployment/ingress/color/image_raw"
    ingress_depth = "/easy_perception_deployment/ingress/aligned_depth/image_raw"
    ingress_info = "/easy_perception_deployment/ingress/color/camera_info"

    contract_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            pkg_share + "/launch/workcell_contract.launch.py"),
        condition=IfCondition(publish_workcell_contract),
        launch_arguments={
            "scene_id": workcell_scene_id,
            "camera_id": workcell_camera_id,
            "profile_ref": perception_profile_ref,
            "runtime_mode": "live",
            "source_mode": contract_source_mode,
            "require_tracking_ids": contract_require_tracking_ids,
        }.items(),
    )

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
        DeclareLaunchArgument(
            "usecase_mode_override",
            default_value="-1",
            description=(
                "Runtime use-case override; -1 preserves the persistent configuration"
            )
        ),
        DeclareLaunchArgument(
            "tracker_type_override",
            default_value="",
            description=(
                "Runtime tracker override; empty preserves the persistent configuration"
            )
        ),
        DeclareLaunchArgument(
            "tracking_maximum_missed_observations",
            default_value="30",
            description=(
                "Number of missed live observations retained before a track expires"
            )
        ),
        DeclareLaunchArgument(
            "publish_workcell_contract",
            default_value="false",
            description=(
                "Publish workcell_perception_snapshot/v1 and connector health. "
                "Enable when Workcell Studio supplies scene/camera identity."
            ),
        ),
        DeclareLaunchArgument(
            "workcell_scene_id",
            default_value="",
            description=(
                "Scene identity supplied by Workcell Studio. EPD does not own or "
                "discover scene definitions."
            ),
        ),
        DeclareLaunchArgument(
            "workcell_camera_id",
            default_value="realsense_d435i_1",
            description="Camera identity expected by the Workcell Studio scene.",
        ),
        DeclareLaunchArgument(
            "perception_profile_ref",
            default_value="",
            description="Optional EPD-5 profile reference stored by the scene.",
        ),
        DeclareLaunchArgument(
            "contract_source_mode",
            default_value="tracking",
            choices=["tracking", "localization", "auto"],
            description=(
                "P3 source for the normalized contract. Tracking is recommended "
                "because it provides stable IDs."
            ),
        ),
        DeclareLaunchArgument(
            "contract_require_tracking_ids",
            default_value="true",
            description=(
                "Reject contract objects without stable Tracking IDs. Set false "
                "only for localization-only workflows."
            ),
        ),

        LogInfo(
            msg=[
                "[epd_emd_pipeline] Runtime overrides: usecase_mode=",
                usecase_mode_override,
                ", tracker_type='",
                tracker_type_override,
                "'. Values -1/empty preserve persistent configuration. "
                "Native EPD outputs remain: localization=/easy_perception_deployment/"
                "epd_localize_output tracking=/easy_perception_deployment/"
                "epd_tracking_output. Workcell contract enabled=",
                publish_workcell_contract,
                " snapshot=/workcell_studio/epd_detection_snapshot_json "
                "status=/workcell_studio/epd_connector_status.",
            ]
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
                {"usecase_mode_override": ParameterValue(
                    usecase_mode_override, value_type=int)},
                {"tracker_type_override": ParameterValue(
                    tracker_type_override, value_type=str)},
                {"tracking_maximum_missed_observations": ParameterValue(
                    tracking_maximum_missed_observations, value_type=int)},
            ],
            remappings=[
                ("/easy_perception_deployment/image_input", ingress_rgb),
            ],
            arguments=["--ros-args", "--log-level", log_level],
        ),
        contract_launch,
    ])
