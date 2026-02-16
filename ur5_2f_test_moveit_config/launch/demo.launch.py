from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def _load_yaml(package_name, relative_path):
    package_path = get_package_share_directory(package_name)
    absolute_path = os.path.join(package_path, relative_path)
    with open(absolute_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_launch_description():
    sensors_3d = _load_yaml('ur5_2f_test_moveit_config', 'config/sensors_3d.yaml')

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            {'octomap_resolution': 0.05},  # explicit value, avoid default fallback 0.1
            {'occupancy_map_monitor': sensors_3d},
        ],
    )

    return LaunchDescription([move_group])
