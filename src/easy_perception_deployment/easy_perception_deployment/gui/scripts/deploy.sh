#!/usr/bin/env bash

msg0="Constructing Docker"
msg1="Sourcing [ROS2]"
msg2="Sourcing [Local Package/Workspace]"
msg3="Deploying package."

useCPU=$1
showImage=$2
ROS_DISTRO="${ROS_DISTRO:-humble}"

if [ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  echo "ROS2 ${ROS_DISTRO} is not installed or setup.bash is missing."
  echo "Please install ROS2 ${ROS_DISTRO} and ensure /opt/ros/${ROS_DISTRO}/setup.bash exists."
  exit 1
fi
source /opt/ros/${ROS_DISTRO}/setup.bash

# Check if Docker is installed.
# If not installed, install it.
if output=$(sudo docker --version > /dev/null 2>&1); then
    :
else
    echo "Installing Docker..."
    echo "Reference: [ https://docs.docker.com/engine/install/ubuntu/ ]"
    sudo apt-get remove -y docker \
      docker-engine \
      docker.io \
      containerd \
      runc
    sudo apt-get update && \
    apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
   sudo apt-get update
   sudo apt-get install docker-ce docker-ce-cli containerd.io
fi

echo "Docker [ FOUND ]"

if [ "$useCPU" = True ] ; then
  # Check if epd-humble-base:CPU Docker Image has NOT been built.
  # If true, build it.
  if output=$(sudo docker images | grep cardboardcode/epd-humble-base | grep CPU > /dev/null 2>&1); then
      echo "epd-humble-base:CPU  Docker Image [ FOUND ]"
  else
      # If there is internet connection,
      # Download public docker image.
      wget -q --spider http://google.com
      if [ $? -eq 0 ]; then
        sudo docker pull cardboardcode/epd-humble-base:CPU
      # Otherwise, build locally
      else
        sudo docker build --tag cardboardcode/epd-humble-base:CPU ../../Dockerfiles/CPU/
      fi
      echo "epd-humble-base:CPU  Docker Image [ CREATED ]"
  fi

else
  # Check if epd-humble-base:GPU Docker Image has NOT been built.
  # If true, build it.
  if output=$(sudo docker images | grep cardboardcode/epd-humble-base | grep GPU > /dev/null 2>&1); then
      echo "epd-humble-base:GPU  Docker Image [ FOUND ]"
  else
      # If there is internet connection,
      # Download public docker image.
      wget -q --spider http://google.com
      if [ $? -eq 0 ]; then
        sudo docker pull cardboardcode/epd-humble-base:GPU
      # Otherwise, build locally
      else
        sudo docker build --tag cardboardcode/epd-humble-base:GPU ../../Dockerfiles/GPU/
      fi
      echo "epd-humble-base:GPU  Docker Image [ CREATED ]"
  fi
fi

# Call ROS2 image_tool showimage.
if [ "$showImage" = True ] ; then
  # Source local ROS2 distro
  source /opt/ros/${ROS_DISTRO}/setup.bash

  ros2 run image_tools showimage --ros-args --remap /image:=/easy_perception_deployment/output > /dev/null 2>&1 &
fi

START_DIR=$(pwd)
cd ../../

read -p "Do you wish to rebuild? [y/n]: " input

if [[ $input == "y" ]]; then
  launch_script="./root/epd_ros2_ws/src/easy_perception_deployment/easy_perception_deployment/gui/scripts/build_launch.sh"
elif [[ $input == "n" ]]; then
  launch_script="./root/epd_ros2_ws/src/easy_perception_deployment/easy_perception_deployment/gui/scripts/launch.sh"
fi

if [ "$useCPU" = True ] ; then
  sudo docker run -it --rm \
  --name epd_test_container \
  -v $(pwd):/root/epd_ros2_ws/src/easy_perception_deployment \
  cardboardcode/epd-humble-base:CPU \
  $launch_script
else
  sudo docker run -it --rm \
  --name epd_test_container \
  -v $(pwd):/root/epd_ros2_ws/src/easy_perception_deployment \
  --gpus all \
  cardboardcode/epd-humble-base:GPU \
  $launch_script
fi

unset input

cd $START_DIR
