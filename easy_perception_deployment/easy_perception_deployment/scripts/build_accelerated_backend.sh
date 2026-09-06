#!/usr/bin/env bash
set -euo pipefail

backend="${1:-cuda}"
case "${backend}" in
  cpu|cuda|tensorrt|jetson) ;;
  *)
    echo "Usage: $0 cpu|cuda|tensorrt|jetson" >&2
    exit 2
    ;;
esac

if [[ -z "${ROS_DISTRO:-}" ]]; then
  if [[ -f /opt/ros/humble/setup.bash ]]; then
    # shellcheck source=/dev/null
    source /opt/ros/humble/setup.bash
  else
    echo "EPD_ERR_ROS_SETUP_MISSING: source ROS 2 Humble before building." >&2
    exit 3
  fi
fi

workspace="${EPD_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
vendor="${workspace}/src/epd_onnxruntime_vendor"
epd="${workspace}/src/easy_perception_deployment"

if [[ ! -d "${vendor}" || ! -d "${epd}" ]]; then
  echo "EPD_ERR_WORKSPACE_LAYOUT: expected epd_onnxruntime_vendor and easy_perception_deployment under ${workspace}/src" >&2
  exit 4
fi

cd "${workspace}"

build_vendor_cpu() {
  colcon build --packages-select epd_onnxruntime_vendor --cmake-clean-cache
}

build_vendor_cuda() {
  if [[ ! -d /usr/local/cuda ]]; then
    echo "EPD_ERR_CUDA_MISSING: /usr/local/cuda is missing." >&2
    exit 5
  fi
  colcon build \
    --packages-select epd_onnxruntime_vendor \
    --cmake-clean-cache \
    --cmake-args -DUSE_CUDA=ON
}

build_vendor_tensorrt() {
  if ! grep -q 'onnxruntime_USE_TENSORRT' "${vendor}/CMakeLists.txt"; then
    cat >&2 <<'EOF'
EPD_ERR_TENSORRT_VENDOR_NOT_READY: the current epd_onnxruntime_vendor only enables CUDA.
Apply the TensorRT vendor extension documented in docs/EPD_PERFORMANCE_BACKENDS.md,
or use a vendor fork that builds ONNX Runtime with onnxruntime_USE_TENSORRT=ON.
EOF
    exit 6
  fi
  if [[ -z "${TENSORRT_HOME:-}" ]]; then
    echo "EPD_ERR_TENSORRT_HOME: set TENSORRT_HOME to the TensorRT installation root." >&2
    exit 6
  fi
  colcon build \
    --packages-select epd_onnxruntime_vendor \
    --cmake-clean-cache \
    --cmake-args \
      -DUSE_CUDA=ON \
      -DUSE_TENSORRT=ON \
      -DTENSORRT_HOME="${TENSORRT_HOME}"
}

case "${backend}" in
  cpu)
    build_vendor_cpu
    ;;
  cuda)
    build_vendor_cuda
    ;;
  jetson)
    if [[ "$(uname -m)" != "aarch64" ]]; then
      echo "EPD_ERR_JETSON_ARCH: jetson backend expects aarch64; current architecture is $(uname -m)." >&2
      exit 7
    fi
    if [[ ! -f /etc/nv_tegra_release && ! -r /proc/device-tree/model ]]; then
      echo "EPD_ERR_JETSON_PLATFORM: NVIDIA Jetson platform markers were not found." >&2
      exit 7
    fi
    build_vendor_cuda
    backend="cuda"
    ;;
  tensorrt)
    build_vendor_tensorrt
    ;;
esac

# shellcheck source=/dev/null
source "${workspace}/install/setup.bash"

if [[ "${backend}" == "tensorrt" ]]; then
  colcon build \
    --packages-select easy_perception_deployment \
    --cmake-clean-cache \
    --cmake-args -DEPD_ENABLE_TENSORRT=ON
else
  colcon build \
    --packages-select easy_perception_deployment \
    --cmake-clean-cache \
    --cmake-args -DEPD_ENABLE_TENSORRT=OFF
fi

cat <<EOF
EPD backend build complete.
Requested backend: ${backend}
Source the workspace before running:
  source ${workspace}/install/setup.bash
Probe compiled capabilities with:
  ros2 run easy_perception_deployment epd_backend_probe
EOF
