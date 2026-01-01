#!/usr/bin/env bash

msg1="Sourcing [ROS2]"
msg2="Sourcing [Local Package/Workspace]"
msg3="Deploying package."

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd "$SCRIPTPATH"

workspace_search_path="$SCRIPTPATH"
WORKSPACE_ROOT=""
while [ "$workspace_search_path" != "/" ]; do
  if [ "$(basename "$workspace_search_path")" = "epd_ros2_ws" ]; then
    WORKSPACE_ROOT="$workspace_search_path"
    break
  fi
  workspace_search_path="$(dirname "$workspace_search_path")"
done

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
if [ -z "$WORKSPACE_ROOT" ]; then
  echo "Unable to locate epd_ros2_ws workspace root from ${SCRIPTPATH}."
  exit 1
fi

cd "$WORKSPACE_ROOT"
if [ -d "build" ] || [ -d "install" ] || [ -d "log" ] ; then
  rm -r build install log
fi
colcon build

if [ -f "${WORKSPACE_ROOT}/install/setup.bash" ]; then
  source "${WORKSPACE_ROOT}/install/setup.bash"
else
  echo "Workspace install/setup.bash not found at ${WORKSPACE_ROOT}/install/setup.bash."
  echo "Please build the workspace before launching."
  exit 1
fi


# Launch easy_perception_deployment.
echo $msg3
ros2 launch easy_perception_deployment run.launch.py
