from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    epd_pkg = get_package_share_directory("easy_perception_deployment")
    ros_gz_sim_pkg = get_package_share_directory("ros_gz_sim")

    camera_model_sdf = (
        "<sdf version='1.9'>"
        "  <model name='epd_rgbd_camera'>"
        "    <pose>0 0 1.2 0 0 0</pose>"
        "    <static>true</static>"
        "    <link name='camera_link'>"
        "      <inertial><mass>0.1</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>"
        "      <visual name='camera_visual'><geometry><box><size>0.1 0.1 0.1</size></box></geometry></visual>"
        "      <sensor name='epd_rgbd' type='rgbd_camera'>"
        "        <always_on>true</always_on>"
        "        <update_rate>15</update_rate>"
        "        <topic>/epd/camera</topic>"
        "        <camera>"
        "          <horizontal_fov>1.047</horizontal_fov>"
        "          <image><width>640</width><height>480</height><format>R8G8B8</format></image>"
        "          <clip><near>0.1</near><far>10.0</far></clip>"
        "        </camera>"
        "      </sensor>"
        "    </link>"
        "  </model>"
        "</sdf>"
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "log_level",
            default_value="info",
            description="EPD node log level",
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([ros_gz_sim_pkg, "launch", "gz_sim.launch.py"])
            ),
            launch_arguments={"gz_args": "-r"}.items(),
        ),

        Node(
            package="ros_gz_sim",
            executable="create",
            output="screen",
            arguments=["-string", camera_model_sdf, "-name", "epd_rgbd_camera"],
        ),

        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            output="screen",
            arguments=[
                "/epd/camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
                "/epd/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
                "/epd/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            ],
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([epd_pkg, "launch", "run.launch.py"])
            ),
            launch_arguments={
                "rgb_topic": "/epd/camera/image",
                "depth_topic": "/epd/camera/depth_image",
                "camera_info_topic": "/epd/camera/camera_info",
                "log_level": LaunchConfiguration("log_level"),
            }.items(),
        ),
    ])
