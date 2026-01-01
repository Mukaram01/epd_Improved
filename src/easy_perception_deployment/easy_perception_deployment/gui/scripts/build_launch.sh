#!/usr/bin/env bash

msg1="Sourcing [ROS2]"
msg2="Sourcing [Local Package/Workspace]"
msg3="Deploying package."

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH

WORKSPACE_ROOT="$(cd "$SCRIPTPATH/../../../../.." >/dev/null 2>&1 ; pwd -P)"
WORKSPACE_SRC="${WORKSPACE_ROOT}/src"
VENDOR_DIR="${WORKSPACE_SRC}/epd_onnxruntime_vendor"
if [ ! -d "$VENDOR_DIR" ]; then
  echo "Missing epd_onnxruntime_vendor at ${VENDOR_DIR}."
  echo "Please ensure epd_onnxruntime_vendor exists in ${WORKSPACE_SRC}."
  exit 1
fi

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
# Build the workspace so vendor packages are available.
cd "$WORKSPACE_ROOT"
if [ -d "build" ] || [ -d "install" ] || [ -d "log" ] ; then
  rm -r build install log
fi
colcon build
source "${WORKSPACE_ROOT}/install/setup.bash"


# Launch easy_perception_deployment.
echo $msg3
ros2 launch easy_perception_deployment run.launch.py
