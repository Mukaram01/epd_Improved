#!/usr/bin/env bash
set -euo pipefail

useCPU="${1:-False}"
showImage="${2:-False}"
shift $(( $# >= 2 ? 2 : $# ))

rebuild="false"
non_interactive="false"
force_sudo_docker="false"
requested_backend="${EPD_DEPLOY_BACKEND:-auto}"
active_backend=""

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
  echo "EPD_DEPLOY_BACKEND may be auto, docker, or local (default: auto)." >&2
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

case "${requested_backend}" in
  auto|docker|local)
    ;;
  *)
    epd_error "EPD_ERR_BAD_BACKEND" "Unsupported EPD_DEPLOY_BACKEND: ${requested_backend}"
    epd_error "EPD_ERR_BAD_BACKEND_REMEDIATION" "Use EPD_DEPLOY_BACKEND=auto, docker, or local."
    exit 2
    ;;
esac

DOCKER_CMD=()
resolve_docker_cmd() {
  if [[ -n "${EPD_DOCKER_CMD:-}" ]]; then
    # shellcheck disable=SC2206
    DOCKER_CMD=(${EPD_DOCKER_CMD})
    [[ ${#DOCKER_CMD[@]} -gt 0 ]]
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

resolve_workspace() {
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

  if [[ "${rebuild}" == "true" ]]; then
    local_launch_script="${SCRIPT_DIR}/build_launch.sh"
  else
    local_launch_script="${SCRIPT_DIR}/launch.sh"
  fi

  container_workspace="/root/epd_ros2_ws"
  container_script_dir="${container_workspace}/${SCRIPT_RELATIVE_PATH}"
  if [[ "${rebuild}" == "true" ]]; then
    container_launch_script="${container_script_dir}/build_launch.sh"
  else
    container_launch_script="${container_script_dir}/launch.sh"
  fi
}

docker_is_available() {
  resolve_docker_cmd || return 1
  if [[ ${#DOCKER_CMD[@]} -eq 0 ]]; then
    return 1
  fi
  if ! command -v "${DOCKER_CMD[-1]}" >/dev/null 2>&1; then
    return 1
  fi
  "${DOCKER_CMD[@]}" --version >/dev/null 2>&1
}

local_runtime_is_available() {
  [[ -x "${local_launch_script}" ]] || return 1
  if [[ "${rebuild}" != "true" && ! -f "${WORKSPACE_ROOT}/install/setup.bash" ]]; then
    return 1
  fi
  return 0
}

resolve_backend() {
  if [[ "${requested_backend}" == "docker" ]]; then
    if ! docker_is_available; then
      epd_error "EPD_ERR_DOCKER_NOT_FOUND" "Docker deployment was requested, but the Docker command is unavailable: ${DOCKER_CMD[*]:-docker}"
      epd_error "EPD_ERR_DOCKER_NOT_FOUND_REMEDIATION" "Install Docker, set EPD_DOCKER_CMD, or use EPD_DEPLOY_BACKEND=local."
      exit 4
    fi
    active_backend="docker"
    return
  fi

  if [[ "${requested_backend}" == "local" ]]; then
    if ! local_runtime_is_available; then
      epd_error "EPD_ERR_LOCAL_RUNTIME_UNAVAILABLE" "Local EPD launch is unavailable. launch=${local_launch_script} workspace=${WORKSPACE_ROOT}"
      epd_error "EPD_ERR_LOCAL_RUNTIME_UNAVAILABLE_REMEDIATION" "Build ${WORKSPACE_ROOT} and ensure the GUI launch script is executable."
      exit 6
    fi
    active_backend="local"
    return
  fi

  # auto: preserve Docker behaviour when it is installed, but do not make
  # Docker a hard requirement for a workspace that is already built locally.
  if docker_is_available; then
    active_backend="docker"
  elif local_runtime_is_available; then
    active_backend="local"
  else
    epd_error "EPD_ERR_DEPLOY_BACKEND_UNAVAILABLE" "Neither Docker nor a built local EPD workspace is available."
    epd_error "EPD_ERR_DEPLOY_BACKEND_UNAVAILABLE_REMEDIATION" "Build ${WORKSPACE_ROOT} or install/configure Docker."
    exit 4
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

start_optional_image_viewer() {
  if [[ "${showImage}" == "True" ]]; then
    ros2 run image_tools showimage --ros-args --remap /image:=/easy_perception_deployment/output >/dev/null 2>&1 &
  fi
}

run_container() {
  local docker_tty=()
  if [[ "${non_interactive}" != "true" && -t 0 ]]; then
    docker_tty=(-t)
  fi

  local vendor_path="${container_workspace}/src/epd_onnxruntime_vendor"
  local container_cmd
  container_cmd="if [ ! -d \"${vendor_path}\" ]; then echo \"EPD_ERR_VENDOR_MISSING: missing ${vendor_path}\" >&2; echo \"EPD_ERR_VENDOR_MISSING_REMEDIATION: mount workspace at ${container_workspace} and include epd_onnxruntime_vendor\" >&2; exit 7; fi; if [ ! -x \"${container_launch_script}\" ]; then echo \"EPD_ERR_LAUNCH_SCRIPT_INVALID: launch script missing or not executable: ${container_launch_script}\" >&2; exit 8; fi; exec \"${container_launch_script}\""

  local docker_args=(run -i "${docker_tty[@]}" --rm --name epd_test_container -v "${WORKSPACE_ROOT}:${container_workspace}")
  if [[ "${useCPU}" != "True" ]]; then
    docker_args+=(--gpus all)
  fi
  docker_args+=("${image_name}" bash -lc "${container_cmd}")

  "${DOCKER_CMD[@]}" "${docker_args[@]}"
}

run_local() {
  if [[ ! -x "${local_launch_script}" ]]; then
    epd_error "EPD_ERR_LAUNCH_SCRIPT_INVALID" "Local launch script missing or not executable: ${local_launch_script}"
    exit 8
  fi
  log_info "Deployment backend [ local workspace ]"
  exec "${local_launch_script}"
}

check_ros
resolve_workspace
resolve_backend
start_optional_image_viewer

if [[ "${active_backend}" == "docker" ]]; then
  log_info "Deployment backend [ Docker ]"
  check_image
  run_container
else
  run_local
fi

cd "${START_DIR}"
