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
if [ -z "$WORKSPACE_ROOT" ]; then
  echo "Unable to locate epd_ros2_ws workspace root from ${SCRIPTPATH}."
  exit 1
fi

if [ -f "${WORKSPACE_ROOT}/install/setup.bash" ]; then
  source "${WORKSPACE_ROOT}/install/setup.bash"
else
  echo "Workspace install/setup.bash not found at ${WORKSPACE_ROOT}/install/setup.bash."
  echo "Please build the workspace before launching."
  exit 1
fi


# Resolve the RGB topic from input_image_topic.json (written by the GUI).
CONFIG_JSON="${SCRIPTPATH}/../../config/input_image_topic.json"
RGB_TOPIC=$(python3 -c "
import json, sys
try:
    with open('${CONFIG_JSON}') as f:
        print(json.load(f).get('input_image_topic', '/camera/camera/color/image_raw'))
except (FileNotFoundError, json.JSONDecodeError, KeyError):
    print('/camera/camera/color/image_raw')
" 2>/dev/null || echo '/camera/camera/color/image_raw')

# Launch easy_perception_deployment.
echo $msg3
ros2 launch easy_perception_deployment run.launch.py rgb_topic:="${RGB_TOPIC}"
