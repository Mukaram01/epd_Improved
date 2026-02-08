#!/usr/bin/env bash
set -euo pipefail

# Copyright 2022 Advanced Remanufacturing and Technology Centre
# Copyright 2022 ROS-Industrial Consortium Asia Pacific Team
# Licensed under the Apache License, Version 2.0

# Function: Installs all dependencies required for EPD to function on CPU mode.
# Static Analysis: shellcheck install_dep_cpu.bash -e SC2086
# Reference: https://github.com/microsoft/onnxruntime#installation

command_exists() { type "$1" &>/dev/null; }

# returns 0 if $1 >= $2
version_greater_equal() { printf '%s\n%s\n' "$2" "$1" | sort -V -C; }

CMAKE_LOWEST_VERSION="3.13"
CMAKE_VERSION="$(cmake --version 2>/dev/null | head -n1 | awk '{print $3}' || true)"

# Check internet early
if wget -q --spider http://google.com; then
  echo "[ WIFI ] - FOUND . Proceeding with download."
else
  echo "[ WIFI ] - NOTFOUND . Please ensure you are properly connected to the internet before running this script again."
  exit 1
fi

echo "-------------------------------------------------------------------------"
echo "Checking dependencies..."
echo "-------------------------------------------------------------------------"

if ! command_exists cmake; then
  echo "CMake not found. Please install CMake ${CMAKE_LOWEST_VERSION}+ first."
  exit 1
fi

if ! version_greater_equal "${CMAKE_VERSION}" "${CMAKE_LOWEST_VERSION}"; then
  echo "Require CMake ${CMAKE_LOWEST_VERSION} or above (found ${CMAKE_VERSION})."
  echo "On Ubuntu 22.04, install with: sudo apt-get install -y cmake"
  exit 1
fi

# Base deps
sudo apt-get update
sudo apt-get install -y wget git python3-rosdep python3-colcon-common-extensions \
  libgomp1 zlib1g-dev locales language-pack-en

sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8

echo "-------------------------------------------------------------------------"
echo "Installing / verifying Anaconda3..."
echo "-------------------------------------------------------------------------"

# If conda missing, install a known Anaconda (kept as per original script)
if conda --version >/dev/null 2>&1; then
  :
else
  cd "$HOME"
  wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
  bash Anaconda3-2020.07-Linux-x86_64.sh -b -p "/home/$USER/anaconda3"
  rm -f Anaconda3-2020.07-Linux-x86_64.sh
  export PATH="/home/$USER/anaconda3/bin:$PATH"
  conda init
  conda config --set auto_activate_base False
  # conda deactivate may fail if not activated; ignore
  conda deactivate >/dev/null 2>&1 || true
fi

echo "-------------------------------------------------------------------------"
echo "Installing onnxruntime (CPU, shared lib, NO tests)..."
echo "-------------------------------------------------------------------------"

cd "$HOME"

ORT_DIR="$HOME/onnxruntime"
ORT_COMMIT="36dc057913f968566eaa1646cb5db41d8c5e7654"
INSTALL_PREFIX="/usr/local"
BUILDTYPE="Release"

if [ ! -d "$ORT_DIR" ]; then
  git clone --recursive https://github.com/microsoft/onnxruntime "$ORT_DIR"
else
  echo "Found an existing onnxruntime at: $ORT_DIR"
  read -r -p "Do you wish to overwrite it [y/n]? " -n 1 REPLY
  echo
  if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    rm -rf "$ORT_DIR"
    git clone --recursive https://github.com/microsoft/onnxruntime "$ORT_DIR"
  fi
fi

cd "$ORT_DIR"
git reset --hard "$ORT_COMMIT"
git submodule sync --recursive
git submodule update --init --recursive

# IMPORTANT:
# - Build as USER (not sudo), otherwise your build dir becomes root-owned.
# - Disable dev/test/report/winml test targets so it won't compile onnxruntime_test_all.
BUILDARGS="--config ${BUILDTYPE}"
BUILDARGS="${BUILDARGS} --parallel"
BUILDARGS="${BUILDARGS} --update"
BUILDARGS="${BUILDARGS} --use_openmp"
BUILDARGS="${BUILDARGS} --skip_tests"
BUILDARGS="${BUILDARGS} --build_shared_lib"
BUILDARGS="${BUILDARGS} --cmake_extra_defines \
CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
onnxruntime_BUILD_SHARED_LIB=ON \
onnxruntime_BUILD_UNIT_TESTS=OFF \
onnxruntime_BUILD_WINML_TESTS=OFF \
onnxruntime_RUN_ONNX_TESTS=OFF \
onnxruntime_GENERATE_TEST_REPORTS=OFF \
onnxruntime_DEV_MODE=OFF"

# If you previously built with sudo, clean/own build dir to avoid permission issues
if [ -d "$ORT_DIR/build" ]; then
  sudo chown -R "$USER:$USER" "$ORT_DIR/build" || true
fi

# Clean build tree to ensure flags take effect
rm -rf "$ORT_DIR/build/Linux/${BUILDTYPE}" || true

# Build (USER)
env "PATH=$PATH" ./build.sh ${BUILDARGS}

# Install (ROOT)
cd "$ORT_DIR/build/Linux/${BUILDTYPE}"
sudo make install

echo
echo "✅ onnxruntime installed to ${INSTALL_PREFIX}"
echo "Uninstall with: cd ${ORT_DIR}/build/Linux/${BUILDTYPE} && cat install_manifest.txt | sudo xargs rm"

