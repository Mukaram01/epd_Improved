#!/usr/bin/env bash
set -euo pipefail

useCPU="${1:-False}"
showImage="${2:-False}"
shift $(( $# >= 2 ? 2 : $# ))

rebuild="false"
non_interactive="false"
force_sudo_docker="false"

log_info() {
  if [[ "${non_interactive}" != "true" ]]; then
    echo "$*"
  fi
}

epd_error() {
  local code="$1"
  shift
  echo "${code}: $*" >&2
}

usage() {
  echo "Usage: $0 <useCPU> <showImage> [--rebuild|--no-rebuild] [--non-interactive] [--sudo-docker]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild)
      rebuild="true"
      ;;
    --no-rebuild)
      rebuild="false"
      ;;
    --non-interactive)
      non_interactive="true"
      ;;
    --sudo-docker)
      force_sudo_docker="true"
      ;;
    *)
      epd_error "EPD_ERR_BAD_OPTION" "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
  shift
done

DOCKER_CMD=()
resolve_docker_cmd() {
  if [[ -n "${EPD_DOCKER_CMD:-}" ]]; then
    # shellcheck disable=SC2206
    DOCKER_CMD=(${EPD_DOCKER_CMD})
    if [[ ${#DOCKER_CMD[@]} -eq 0 ]]; then
      epd_error "EPD_ERR_DOCKER_CMD_INVALID" "EPD_DOCKER_CMD is set but empty."
      exit 4
    fi
    return
  fi

  if [[ "${force_sudo_docker}" == "true" ]]; then
    DOCKER_CMD=(sudo docker)
    return
  fi

  DOCKER_CMD=(docker)
}

check_ros() {
  local ros_distro ros_setup
  ros_distro="${ROS_DISTRO:-humble}"
  ros_setup="/opt/ros/${ros_distro}/setup.bash"

  if [[ ! -f "${ros_setup}" ]]; then
    epd_error "EPD_ERR_ROS_SETUP_MISSING" "Missing ROS setup file: ${ros_setup}"
    epd_error "EPD_ERR_ROS_SETUP_MISSING_REMEDIATION" "Install ROS 2 ${ros_distro} and ensure setup.bash exists."
    exit 3
  fi

  set +u
  : "${AMENT_TRACE_SETUP_FILES:=}"
  # shellcheck source=/dev/null
  source "${ros_setup}"
  set -u
}

check_docker() {
  resolve_docker_cmd

  if ! command -v "${DOCKER_CMD[-1]}" >/dev/null 2>&1; then
    epd_error "EPD_ERR_DOCKER_NOT_FOUND" "Docker command not found: ${DOCKER_CMD[*]}"
    epd_error "EPD_ERR_DOCKER_NOT_FOUND_REMEDIATION" "Install Docker or set EPD_DOCKER_CMD to a valid command."
    exit 4
  fi

  if ! "${DOCKER_CMD[@]}" --version >/dev/null 2>&1; then
    epd_error "EPD_ERR_DOCKER_UNAVAILABLE" "Docker is not installed or not accessible via: ${DOCKER_CMD[*]}"
    epd_error "EPD_ERR_DOCKER_UNAVAILABLE_REMEDIATION" "Install Docker, grant user access to docker group, or set EPD_DOCKER_CMD/--sudo-docker explicitly."
    exit 4
  fi

  log_info "Docker command [ ${DOCKER_CMD[*]} ]"
}

check_workspace_mount() {
  START_DIR="$(pwd)"
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  DEFAULT_WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
  WORKSPACE_ROOT="${EPD_WORKSPACE_ROOT:-${DEFAULT_WORKSPACE_ROOT}}"

  if [[ ! -d "${WORKSPACE_ROOT}" ]]; then
    epd_error "EPD_ERR_WORKSPACE_MISSING" "Workspace root does not exist: ${WORKSPACE_ROOT}"
    exit 6
  fi

  SCRIPT_RELATIVE_PATH="${SCRIPT_DIR#${WORKSPACE_ROOT}/}"
  if [[ -z "${SCRIPT_RELATIVE_PATH}" || "${SCRIPT_RELATIVE_PATH}" == "${SCRIPT_DIR}" ]]; then
    epd_error "EPD_ERR_WORKSPACE_LAYOUT" "script dir not under workspace root. script_dir=${SCRIPT_DIR} workspace_root=${WORKSPACE_ROOT}"
    exit 6
  fi

  container_workspace="/root/epd_ros2_ws"
  container_script_dir="${container_workspace}/${SCRIPT_RELATIVE_PATH}"
  if [[ "${rebuild}" == "true" ]]; then
    launch_script="${container_script_dir}/build_launch.sh"
  else
    launch_script="${container_script_dir}/launch.sh"
  fi
}

check_image() {
  if [[ "${useCPU}" == "True" ]]; then
    image_name="cardboardcode/epd-humble-base:CPU"
    local_build_dir="../../Dockerfiles/CPU/"
  else
    image_name="cardboardcode/epd-humble-base:GPU"
    local_build_dir="../../Dockerfiles/GPU/"
  fi

  if ! "${DOCKER_CMD[@]}" image inspect "${image_name}" >/dev/null 2>&1; then
    epd_error "EPD_ERR_IMAGE_NOT_FOUND" "Docker image not found: ${image_name}"
    epd_error "EPD_ERR_IMAGE_NOT_FOUND_REMEDIATION" "${DOCKER_CMD[*]} pull ${image_name} OR ${DOCKER_CMD[*]} build --tag ${image_name} ${local_build_dir}"
    exit 5
  fi

  log_info "${image_name} Docker Image [ FOUND ]"
}

run_container() {
  local docker_tty=()
  if [[ "${non_interactive}" != "true" && -t 0 ]]; then
    docker_tty=(-t)
  fi

  local vendor_path="${container_workspace}/src/epd_onnxruntime_vendor"
  local container_cmd
  container_cmd="if [ ! -d \"${vendor_path}\" ]; then echo \"EPD_ERR_VENDOR_MISSING: missing ${vendor_path}\" >&2; echo \"EPD_ERR_VENDOR_MISSING_REMEDIATION: mount workspace at ${container_workspace} and include epd_onnxruntime_vendor\" >&2; exit 7; fi; if [ ! -x \"${launch_script}\" ]; then echo \"EPD_ERR_LAUNCH_SCRIPT_INVALID: launch script missing or not executable: ${launch_script}\" >&2; exit 8; fi; exec \"${launch_script}\""

  local docker_args=(run -i "${docker_tty[@]}" --rm --name epd_test_container -v "${WORKSPACE_ROOT}:${container_workspace}")
  if [[ "${useCPU}" != "True" ]]; then
    docker_args+=(--gpus all)
  fi
  docker_args+=("${image_name}" bash -lc "${container_cmd}")

  "${DOCKER_CMD[@]}" "${docker_args[@]}"
}

check_ros
check_docker
check_workspace_mount
check_image

if [[ "${showImage}" == "True" ]]; then
  ros2 run image_tools showimage --ros-args --remap /image:=/easy_perception_deployment/output >/dev/null 2>&1 &
fi

run_container

cd "${START_DIR}"
