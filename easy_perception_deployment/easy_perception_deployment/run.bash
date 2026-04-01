#!/usr/bin/env bash
set -euo pipefail

# Init conda (do NOT prepend base anaconda bin to PATH)
# set +u: conda.sh references PS1 which is unset in non-interactive shells
set +u
: "${PS1:=}"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
set -u

# export PATH=~/anaconda3/bin:$PATH
PATH_TO_THIS_SCRIPT=$(realpath "$0")
START_DIR=$(dirname "$PATH_TO_THIS_SCRIPT")

cd "$START_DIR"

EPD_SKIP_DOWNLOAD="${EPD_SKIP_DOWNLOAD:-0}"
EPD_WS="${EPD_WS:-}"

# Check if Anaconda has been installed in general.
# If true, get the first digit of the string which should reflect the major version of conda.
if detected_conda=$(conda --version); then
    # echo $detected_conda - FOUND
    declare -i conda_ver
    conda_ver=$(echo "$detected_conda" | grep -o -E '[0-9]+' | head -1 | sed -e 's/^0\+//')
else
    echo "Please install Anaconda by refering to the installation docs."
    echo "Exiting terminal in 10 seconds."
    echo "[ https://docs.anaconda.com/anaconda/install/linux/ ]"
    sleep 10
    exit 1
fi

# Check if Anaconda is Anaconda2 or below.
if (( conda_ver < 2 )); then
    echo "Anaconda3 - NOTFOUND. Please install Anaconda3."
    echo "Exiting terminal in 10 seconds."
    sleep 10
    exit 1
fi

# Check if pretrained models have been downloaded.
verify_sha256() {
    local file_path=$1
    local expected_hash=$2
    local actual_hash=""

    if command -v sha256sum >/dev/null 2>&1; then
        actual_hash=$(sha256sum "$file_path" | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
        actual_hash=$(shasum -a 256 "$file_path" | awk '{print $1}')
    else
        echo "Warning: sha256sum/shasum not found; skipping hash verification for $file_path."
        return 0
    fi

    if [ "$actual_hash" != "$expected_hash" ]; then
        echo "Checksum mismatch for $file_path."
        echo "Expected SHA256: $expected_hash"
        echo "Actual SHA256:   $actual_hash"
        echo "Deleting corrupted file. Please re-run to download again."
        rm -f "$file_path"
        exit 1
    fi
}

ensure_model() {
    local file_path=$1
    local url=$2
    local expected_hash="${3:-}"

    if [ ! -f "$file_path" ]; then
        if [ "$EPD_SKIP_DOWNLOAD" = "1" ]; then
            echo "EPD_SKIP_DOWNLOAD=1: would download $url -> $file_path"
        else
            echo "Downloading $file_path."
            wget -O "$file_path" "$url"
        fi
    fi

    if [ -n "$expected_hash" ] && [ -f "$file_path" ]; then
        verify_sha256 "$file_path" "$expected_hash"
    fi
}

ensure_model \
    ./data/model/FasterRCNN-10.onnx \
    "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx" \
    dfb81423efbea52e45df242ade64cfca0ba05fae78e00cf0c68a0979987f87eb

ensure_model \
    ./data/model/MaskRCNN-10.onnx \
    "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx" \
    a519d8102cb162e78cbf123615aa5a8f3bf9d0fa1dec61a2fbbb42fa3f0e0757

ensure_model \
    ./data/model/ssd_mobilenet_v1_12.onnx \
    "https://huggingface.co/onnxmodelzoo/ssd_mobilenet_v1_12/resolve/main/ssd_mobilenet_v1_12.onnx?download=true"

# Checking if the epd_gui_env conda environment has been installed.
env_exists=$(conda env list | grep epd_gui_env || true)

if [ -z "$env_exists" ]
then
      echo "Installing epd_gui_env conda environment."
      conda create -n epd_gui_env python=3.10 -y
      conda activate epd_gui_env
      pip install PySide6
      pip install dateutils==0.6.12
      pip install "pycocotools>=2.0.6"
      pip install labelme==5.0.1
      conda deactivate
      echo "[epd_gui_env] env created."
fi

resolve_epd_workspace_setup() {
    local ws="${EPD_WS}"

    if [ -n "$ws" ]; then
        echo "${ws%/}/install/setup.bash"
        return 0
    fi

    if [ -n "${COLCON_PREFIX_PATH:-}" ]; then
        local first_prefix
        IFS=':' read -r first_prefix _ <<< "$COLCON_PREFIX_PATH"
        if [ -n "$first_prefix" ]; then
            echo "${first_prefix%/}/setup.bash"
            return 0
        fi
    fi

    echo "$HOME/epd_ros2_ws/install/setup.bash"
}

# Check for libxcb-cursor0, required by the Qt xcb platform plugin since Qt 6.5.0.
if command -v dpkg >/dev/null 2>&1 && ! dpkg -l libxcb-cursor0 2>/dev/null | grep -q "^ii"; then
    echo "Missing dependency: libxcb-cursor0 (required by Qt xcb platform plugin since Qt 6.5.0)."
    echo "Installing libxcb-cursor0..."
    sudo apt-get install -y libxcb-cursor0 || {
        echo "ERROR: Could not install libxcb-cursor0 automatically."
        echo "Please run manually: sudo apt-get install -y libxcb-cursor0"
        exit 1
    }
fi

conda activate epd_gui_env

# Ensure PySide6 is installed in the active environment.
# Older epd_gui_env installations may only have PySide2; installing PySide6
# here is a no-op when it is already present.
if ! python -c "import PySide6" 2>/dev/null; then
    echo "PySide6 not found in epd_gui_env; installing..."
    pip install PySide6
fi

ROS_DISTRO="${ROS_DISTRO:-humble}"

# ---- FIX: ROS setup + `set -u` compatibility (prevents AMENT_TRACE_SETUP_FILES unbound)
set +u
: "${AMENT_TRACE_SETUP_FILES:=}"
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

EPD_WS_SETUP=$(resolve_epd_workspace_setup)
if [ ! -f "$EPD_WS_SETUP" ]; then
    echo "Unable to find workspace setup script: $EPD_WS_SETUP"
    echo "Set EPD_WS to your workspace root or COLCON_PREFIX_PATH to an installed prefix."
    exit 1
fi

set +u
: "${AMENT_TRACE_SETUP_FILES:=}"
source "$EPD_WS_SETUP"
set -u
# ---- end fix

cd "$START_DIR/gui"
python main.py

unset START_DIR PATH_TO_THIS_SCRIPT env_exists EPD_WS_SETUP EPD_SKIP_DOWNLOAD EPD_WS

