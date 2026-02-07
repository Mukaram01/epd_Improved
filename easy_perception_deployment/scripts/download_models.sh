#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/easy_perception_deployment/data/model"

SQUEEZENET_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
FASTER_RCNN_URL="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx"
MASK_RCNN_URL="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx"

mkdir -p "${MODEL_DIR}"
curl -L "${SQUEEZENET_URL}" -o "${MODEL_DIR}/squeezenet1.1-7.onnx"
curl -L "${FASTER_RCNN_URL}" -o "${MODEL_DIR}/FasterRCNN-10.onnx"
curl -L "${MASK_RCNN_URL}" -o "${MODEL_DIR}/MaskRCNN-10.onnx"
