#!/usr/bin/env bash
set -euo pipefail

msg0="Constructing Docker"
msg1="Sourcing [ROS2]"
msg2="Sourcing [Local Package/Workspace]"
msg3="Deploying package."

useCPU="${1:-False}"
showImage="${2:-False}"
shift $(( $# >= 2 ? 2 : $# ))

rebuild="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild)
      rebuild="true"
      ;;
    --no-rebuild)
      rebuild="false"
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      echo "Usage: $0 <useCPU> <showImage> [--rebuild|--no-rebuild]" >&2
      exit 2
      ;;
  esac
  shift
done

ROS_DISTRO="${ROS_DISTRO:-humble}"
ros_setup="/opt/ros/${ROS_DISTRO}/setup.bash"

if [[ ! -f "${ros_setup}" ]]; then
  echo "ERROR: Missing ROS setup file: ${ros_setup}" >&2
  echo "Install ROS 2 ${ROS_DISTRO} and ensure setup.bash exists." >&2
  exit 3
fi
source "${ros_setup}"

if ! sudo docker --version >/dev/null 2>&1; then
  echo "ERROR: Docker is not installed or not accessible via sudo." >&2
  echo "Remediation: Install Docker, then retry." >&2
  echo "  See: https://docs.docker.com/engine/install/ubuntu/" >&2
  exit 4
fi

echo "Docker [ FOUND ]"

if [[ "${useCPU}" == "True" ]]; then
  image_name="cardboardcode/epd-humble-base:CPU"
  local_build_dir="../../Dockerfiles/CPU/"
else
  image_name="cardboardcode/epd-humble-base:GPU"
  local_build_dir="../../Dockerfiles/GPU/"
fi

if ! sudo docker image inspect "${image_name}" >/dev/null 2>&1; then
  echo "ERROR: Docker image not found: ${image_name}" >&2
  echo "Remediation (choose one):" >&2
  echo "  sudo docker pull ${image_name}" >&2
  echo "  sudo docker build --tag ${image_name} ${local_build_dir}" >&2
  exit 5
fi

echo "${image_name} Docker Image [ FOUND ]"

if [[ "${showImage}" == "True" ]]; then
  ros2 run image_tools showimage --ros-args --remap /image:=/easy_perception_deployment/output >/dev/null 2>&1 &
fi

START_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
WORKSPACE_ROOT="${EPD_WORKSPACE_ROOT:-${DEFAULT_WORKSPACE_ROOT}}"
SCRIPT_RELATIVE_PATH="${SCRIPT_DIR#${WORKSPACE_ROOT}/}"
if [[ -z "${SCRIPT_RELATIVE_PATH}" || "${SCRIPT_RELATIVE_PATH}" == "${SCRIPT_DIR}" ]]; then
  echo "ERROR: script dir not under workspace root." >&2
  echo "script_dir=${SCRIPT_DIR}" >&2
  echo "workspace_root=${WORKSPACE_ROOT}" >&2
  exit 6
fi

container_workspace="/root/epd_ros2_ws"
container_script_dir="${container_workspace}/${SCRIPT_RELATIVE_PATH}"
if [[ "${rebuild}" == "true" ]]; then
  launch_script="${container_script_dir}/build_launch.sh"
else
  launch_script="${container_script_dir}/launch.sh"
fi

docker_tty=()
if [[ -t 0 ]]; then
  docker_tty=(-t)
fi
vendor_path="${container_workspace}/src/epd_onnxruntime_vendor"
container_cmd="if [ ! -d \"${vendor_path}\" ]; then echo \"ERROR: missing ${vendor_path}\" >&2; echo \"Remediation: mount workspace at ${container_workspace} and include epd_onnxruntime_vendor\" >&2; exit 7; fi; if [ ! -x \"${launch_script}\" ]; then echo \"ERROR: launch script missing or not executable: ${launch_script}\" >&2; exit 8; fi; exec \"${launch_script}\""

if [[ "${useCPU}" == "True" ]]; then
  sudo docker run -i "${docker_tty[@]}" --rm \
    --name epd_test_container \
    -v "${WORKSPACE_ROOT}:${container_workspace}" \
    "${image_name}" \
    bash -lc "${container_cmd}"
else
  sudo docker run -i "${docker_tty[@]}" --rm \
    --name epd_test_container \
    -v "${WORKSPACE_ROOT}:${container_workspace}" \
    --gpus all \
    "${image_name}" \
    bash -lc "${container_cmd}"
fi

cd "${START_DIR}"
