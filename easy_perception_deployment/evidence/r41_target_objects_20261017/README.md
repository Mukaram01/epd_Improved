# R4.1 target-object selection — 17 October 2026

## What the D435i evidence showed

`../real_camera_integration_20261016/frames.jsonl` records, per frame and mode,
the objects EPD actually published on the valid-pose outputs (12 real D435i
frames, MaskRCNN-10). Before this change Localization/Tracking had no class
filter, so non-target objects reached EMD as valid poses:

| Mode | Published | Non-target published |
|---|---:|---|
| default (legacy localization) | 40 | mouse 3, bicycle 1, bottle 3 |
| hybrid | 28 | mouse 3, bicycle 1, bottle 3 |
| legacy_tracking | 40 | mouse 3, bicycle 1, bottle 3 |
| hybrid_tracking | 28 | mouse 3, bicycle 1, bottle 3 |

The bicycle and bottle proposals are false detections (see that folder's README).

## With `class_list: ["banana", "apple"]`

Applying `EPD::isClassSelected` (exact label match; empty list = all classes)
to the same recorded outputs: all 33/21 banana+apple objects are kept, and all
7 non-target objects per mode are withheld from the valid-pose outputs. In the
runtime they stay in `/easy_perception_deployment/detection_readiness` with
note `not_selected`, never `GRASP_CANDIDATE`.

**Scope of this evidence:** the rule was applied offline to recorded outputs.
It is not a runtime replay: the raw frames and the model are outside Git, and the
container used for R4.1 has no ROS 2/OpenCV/ONNX Runtime. Live D435i acceptance
is **BLOCKED** until the workstation run below.

## Workstation acceptance (Ubuntu 22.04, ROS 2 Humble, D435i)

```bash
cd ~/epd_ros2_ws && source /opt/ros/humble/setup.bash
colcon build --packages-up-to easy_perception_deployment
colcon test --packages-select easy_perception_deployment \
  --ctest-args -R 'epd_test_(class_selection|perception_readiness)'
source install/setup.bash
# GUI: Deploy > Mode Localization > Objects > tick banana + apple > Run Perception.
# Expect usecase_config.json: {"usecase_mode": 3, "class_list": ["banana", "apple"]}
ros2 topic echo /easy_perception_deployment/epd_localize_output --field objects | grep 'name:'
ros2 topic echo /easy_perception_deployment/detection_readiness
```

Pass: only banana/apple appear on the localize/tracking outputs; any mouse or
other detection appears in readiness with `note: not_selected` and is not
`GRASP_CANDIDATE`; clearing the selection ("All classes") restores the
previous behaviour.

## Screenshots (offscreen render of the full `main.py` GUI stack)

- `deploy_objects_banana_apple.png`: Deploy with Localization and the new Objects row.
- `target_object_picker.png`: searchable picker built from the active label list.
- `launcher.png`: launcher (fixed "Help & Guides" label).
