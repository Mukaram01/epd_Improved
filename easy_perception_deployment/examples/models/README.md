# EPD Reference Model Examples

EPD deployment uses ONNX models. This directory documents reference model choices without committing large model binaries into the release-evidence layer.

The package CMake configuration already recognizes these example assets under `data/model/`:

| Model | Typical purpose | Example labels |
|---|---|---|
| `squeezenet1.1-7.onnx` | image classification | ImageNet labels |
| `FasterRCNN-10.onnx` | bounding-box object detection | COCO labels |
| `MaskRCNN-10.onnx` | instance segmentation / 3D manipulation starting point | COCO labels |
| `ssd_mobilenet_v1_12.onnx` | lighter object detection | matching model labels |

Model downloads are disabled by default during configure/build for reproducible/offline-friendly ROS build behavior. Use the repository's documented model-download path or provide your own validated ONNX model.

Before promoting any model into a release profile:

1. verify the file hash/source;
2. inspect it with the EPD-3 Smart Model Manager;
3. select the exact matching label list;
4. confirm the intended perception mode;
5. run live camera validation;
6. run deterministic replay where the model is compatible with the fixture;
7. capture a target-machine EPD-5 profile so the actual model/label SHA256 values become part of the handoff evidence.

Do not infer that an arbitrary ONNX file is EPD-compatible only because ONNX Runtime can open it. EPD model I/O/post-processing expectations still need to match the selected pipeline.
