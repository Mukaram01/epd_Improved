#!/usr/bin/env bash
set -euo pipefail

# Legacy positional arguments are retained for old launchers. EPD-8 prefers the
# explicit execution_backend stored in config/session_config.json or --backend.
useCPU="${1:-False}"
showImage="${2:-False}"
shift $(( $# >= 2 ? 2 : $# ))

rebuild="false"
non_interactive="false"
force_sudo_docker="false"
backend_override=""
gpu_index_override=""

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
  echo "Usage: $0 <useCPU> <showImage> [--backend auto|cpu|cuda|tensorrt] [--gpu-index N] [--rebuild|--no-rebuild] [--non-interactive] [--sudo-docker]" >&2
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
    --backend)
      shift
      [[ $# -gt 0 ]] || { epd_error "EPD_ERR_BAD_OPTION" "--backend requires a value"; exit 2; }
      backend_override="$1"
      ;;
    --gpu-index)
      shift
      [[ $# -gt 0 ]] || { epd_error "EPD_ERR_BAD_OPTION" "--gpu-index requires a value"; exit 2; }
      gpu_index_override="$1"
      ;;
    *)
      epd_error "EPD_ERR_BAD_OPTION" "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_CONFIG="${SCRIPT_DIR}/../../config/session_config.json"

read_backend_config() {
  local legacy_backend
  legacy_backend="cuda"
  if [[ "${useCPU}" == "True" ]]; then
    legacy_backend="cpu"
  fi

  local config_backend="${legacy_backend}"
  local config_gpu_index="0"
  if [[ -f "${SESSION_CONFIG}" ]] && command -v python3 >/dev/null 2>&1; then
    local values
    values="$(python3 - "${SESSION_CONFIG}" "${legacy_backend}" <<'PY'
import json
import sys
path, legacy = sys.argv[1], sys.argv[2]
try:
    with open(path, encoding="utf-8") as stream:
        cfg = json.load(stream)
except Exception:
    cfg = {}
backend = str(cfg.get("execution_backend", legacy) or legacy).strip().lower()
aliases = {"gpu": "cuda", "nvidia": "cuda", "trt": "tensorrt", "default": "auto"}
backend = aliases.get(backend, backend)
try:
    gpu_index = max(0, int(cfg.get("execution_backend_gpu_index", 0)))
except Exception:
    gpu_index = 0
print(backend)
print(gpu_index)
PY
)"
    config_backend="$(printf '%s\n' "${values}" | sed -n '1p')"
    config_gpu_index="$(printf '%s\n' "${values}" | sed -n '2p')"
  fi

  requested_backend="${backend_override:-${config_backend}}"
  requested_backend="$(printf '%s' "${requested_backend}" | tr '[:upper:]' '[:lower:]')"
  case "${requested_backend}" in
    gpu|nvidia) requested_backend="cuda" ;;
    trt) requested_backend="tensorrt" ;;
    default) requested_backend="auto" ;;
  esac
  case "${requested_backend}" in
    auto|cpu|cuda|tensorrt) ;;
    *)
      epd_error "EPD_ERR_BACKEND_INVALID" "Unsupported execution backend: ${requested_backend}"
      epd_error "EPD_ERR_BACKEND_INVALID_REMEDIATION" "Choose auto, cpu, cuda, or tensorrt."
      exit 9
      ;;
  esac

  gpu_index="${gpu_index_override:-${config_gpu_index}}"
  if ! [[ "${gpu_index}" =~ ^[0-9]+$ ]]; then
    epd_error "EPD_ERR_GPU_INDEX_INVALID" "GPU index must be a non-negative integer: ${gpu_index}"
    exit 9
  fi
}

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

is_jetson() {
  if [[ -f /etc/nv_tegra_release ]]; then
    return 0
  fi
  if [[ "$(uname -m)" == "aarch64" ]] && [[ -r /proc/device-tree/model ]]; then
    grep -qi 'nvidia\|jetson' /proc/device-tree/model 2>/dev/null
    return $?
  fi
  return 1
}

has_nvidia_container_runtime() {
  local runtimes
  runtimes="$("${DOCKER_CMD[@]}" info --format '{{json .Runtimes}}' 2>/dev/null || true)"
  if grep -qi 'nvidia' <<<"${runtimes}"; then
    return 0
  fi
  # New NVIDIA Container Toolkit installations can support --gpus even when
  # the runtime list is not explicit. A working nvidia-smi is useful evidence.
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    return 0
  fi
  is_jetson
}

check_workspace_mount() {
  START_DIR="$(pwd)"
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

select_backend_and_image() {
  cpu_image="${EPD_CPU_IMAGE:-cardboardcode/epd-humble-base:CPU}"
  gpu_image="${EPD_GPU_IMAGE:-cardboardcode/epd-humble-base:GPU}"
  tensorrt_image="${EPD_TENSORRT_IMAGE:-}"
  runtime_backend="${requested_backend}"

  if [[ "${requested_backend}" == "auto" ]]; then
    if has_nvidia_container_runtime; then
      runtime_backend="cuda"
    else
      runtime_backend="cpu"
    fi
  fi

  case "${runtime_backend}" in
    cpu)
      image_name="${cpu_image}"
      local_build_dir="../../Dockerfiles/CPU/"
      ;;
    cuda)
      if ! has_nvidia_container_runtime; then
        if [[ "${requested_backend}" == "auto" ]]; then
          runtime_backend="cpu"
          image_name="${cpu_image}"
          local_build_dir="../../Dockerfiles/CPU/"
        else
          epd_error "EPD_ERR_CUDA_RUNTIME_UNAVAILABLE" "CUDA backend requested but NVIDIA Docker runtime/GPU was not detected."
          epd_error "EPD_ERR_CUDA_RUNTIME_UNAVAILABLE_REMEDIATION" "Install NVIDIA Container Toolkit, verify the GPU, or select CPU/auto."
          exit 10
        fi
      else
        image_name="${gpu_image}"
        local_build_dir="../../Dockerfiles/GPU/"
      fi
      ;;
    tensorrt)
      if ! has_nvidia_container_runtime; then
        epd_error "EPD_ERR_TENSORRT_RUNTIME_UNAVAILABLE" "TensorRT requires an NVIDIA GPU runtime."
        exit 10
      fi
      if [[ -z "${tensorrt_image}" ]]; then
        epd_error "EPD_ERR_TENSORRT_IMAGE_REQUIRED" "TensorRT is opt-in and requires EPD_TENSORRT_IMAGE."
        epd_error "EPD_ERR_TENSORRT_IMAGE_REQUIRED_REMEDIATION" "Use an image whose epd_onnxruntime_vendor was built with TensorRT and build EPD with -DEPD_ENABLE_TENSORRT=ON."
        exit 10
      fi
      image_name="${tensorrt_image}"
      local_build_dir=""
      ;;
  esac

  if ! "${DOCKER_CMD[@]}" image inspect "${image_name}" >/dev/null 2>&1; then
    if [[ "${requested_backend}" == "auto" && "${runtime_backend}" == "cuda" ]]; then
      log_info "GPU image unavailable; AUTO is falling back to CPU."
      runtime_backend="cpu"
      image_name="${cpu_image}"
      local_build_dir="../../Dockerfiles/CPU/"
    fi
  fi
}

check_image() {
  if ! "${DOCKER_CMD[@]}" image inspect "${image_name}" >/dev/null 2>&1; then
    epd_error "EPD_ERR_IMAGE_NOT_FOUND" "Docker image not found for ${runtime_backend}: ${image_name}"
    if [[ -n "${local_build_dir}" ]]; then
      epd_error "EPD_ERR_IMAGE_NOT_FOUND_REMEDIATION" "${DOCKER_CMD[*]} pull ${image_name} OR ${DOCKER_CMD[*]} build --tag ${image_name} ${local_build_dir}"
    else
      epd_error "EPD_ERR_IMAGE_NOT_FOUND_REMEDIATION" "Build/provide a compatible TensorRT image and set EPD_TENSORRT_IMAGE."
    fi
    exit 5
  fi

  log_info "Backend [ ${runtime_backend^^} ]  GPU index [ ${gpu_index} ]"
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

  local docker_args=(
    run -i "${docker_tty[@]}" --rm --name epd_test_container
    -v "${WORKSPACE_ROOT}:${container_workspace}"
    -e "EPD_EXECUTION_BACKEND=${runtime_backend}"
    -e "EPD_GPU_INDEX=${gpu_index}"
  )
  if [[ "${runtime_backend}" != "cpu" ]]; then
    if is_jetson; then
      docker_args+=(--runtime nvidia)
    else
      docker_args+=(--gpus all)
    fi
  fi
  docker_args+=("${image_name}" bash -lc "${container_cmd}")

  "${DOCKER_CMD[@]}" "${docker_args[@]}"
}

read_backend_config
check_ros
check_docker
check_workspace_mount
select_backend_and_image
check_image

if [[ "${showImage}" == "True" ]]; then
  ros2 run image_tools showimage --ros-args --remap /image:=/easy_perception_deployment/output >/dev/null 2>&1 &
fi

run_container

cd "${START_DIR}"
