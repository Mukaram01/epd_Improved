# EPD Example Profiles

These files are portable starting templates for EPD-5 Profiles & Replay.

They are **not** pre-certified or automatically known-good. A template should only be promoted to known-good after the exact target workstation, camera, model, labels and backend have passed the release acceptance checklist.

## Reference RealSense Tracking CPU

`realsense_tracking_cpu.epd-profile.json` uses:

- Mask R-CNN ONNX path: `./data/model/MaskRCNN-10.onnx`
- COCO labels: `./data/label_list/coco_classes.txt`
- Tracking + MEDIANFLOW
- RealSense RGB topic: `/camera/camera/color/image_raw`
- CPU backend
- raw image transport
- detection overlay enabled
- object masks enabled
- confidence threshold 0.50

The template deliberately leaves asset SHA256 values empty because the repository may not contain downloaded model files at source-control time. When you save a profile from the EPD GUI on a configured workstation, EPD-5 captures hashes for assets that exist.

For release evidence, prefer a newly captured target-machine profile with actual asset hashes over this template.
