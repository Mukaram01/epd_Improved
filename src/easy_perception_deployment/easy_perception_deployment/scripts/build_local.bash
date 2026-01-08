#!/usr/bin/env bash

# Copyright 2022 Advanced Remanufacturing and Technology Centre
# Copyright 2022 ROS-Industrial Consortium Asia Pacific Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Function: Source all required setup.bash and build both epd_msgs easy_perception_deployment
# Static Analysis: shellcheck build_local.bash -x -e SC1091

# Sourcing [ ROS2 ]
ROS_DISTRO="${ROS_DISTRO:-humble}"
if [ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  echo "ROS2 ${ROS_DISTRO} is not installed or setup.bash is missing."
  echo "Please install ROS2 ${ROS_DISTRO} and ensure /opt/ros/${ROS_DISTRO}/setup.bash exists."
  exit 1
else
  source /opt/ros/${ROS_DISTRO}/setup.bash
fi

# Build epd_msgs ROS2 package.

if [ ! -d "../epd_msgs/" ]; then
  echo "[ epd_msgs ] ROS2 package is missing."
  exit 1
else
  cd ../epd_msgs/ || exit
  if [ -d  "build/" ]; then
    sudo rm -rf build/
  fi
  if [ -d  "install/" ]; then
    sudo rm -rf install/
  fi
  if [ -d  "log/" ]; then
    sudo rm -rf log/
  fi
  echo "Building and Sourcing [ epd_msgs ]"
  if ! colcon build; then
    echo "Error: colcon build failed for [ epd_msgs ]."
    exit 1
  fi
  source install/setup.bash
fi

# Build easy_perception_deployment ROS2 package.
cd ../easy_perception_deployment/ || exit
if [ -d  "build/" ]; then
  sudo rm -rf build/
fi
if [ -d  "install/" ]; then
  sudo rm -rf install/
fi
if [ -d  "log/" ]; then
  sudo rm -rf log/
fi
echo "Building and Sourcing [ easy_perception_deployment ]"
if ! colcon build; then
  echo "Error: colcon build failed for [ easy_perception_deployment ]."
  exit 1
fi
source install/setup.bash
