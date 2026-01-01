#!/usr/bin/env bash

msg1="Sourcing [ROS2]"
msg2="Sourcing [Local Package/Workspace]"
msg3="Deploying package."

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH

# Source ROS distro
ROS_DISTRO="${ROS_DISTRO:-humble}"
if [ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  echo "ROS2 ${ROS_DISTRO} is not installed or setup.bash is missing."
  echo "Please install ROS2 ${ROS_DISTRO} and ensure /opt/ros/${ROS_DISTRO}/setup.bash exists."
  exit 1
fi
echo $msg1
source /opt/ros/${ROS_DISTRO}/setup.bash

echo $msg2
# Check if the current easy_perception workspace has been built or not.
# If true, run selective colcon build.
# Otherwise, pass
cd ../../
source install/setup.bash

# Source the main workspace
cd ../easy_perception_deployment
source install/setup.bash


# Launch easy_perception_deployment.
echo $msg3
ros2 launch easy_perception_deployment run.launch.py
