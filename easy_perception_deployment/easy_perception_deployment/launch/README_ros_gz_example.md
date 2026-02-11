# Gazebo Sim (ros_gz) EPD example

This example runs a **simulation-first** pipeline with Gazebo Sim (`ros_gz_sim`) and EPD, without real camera hardware.

## What it launches

`ros2 launch easy_perception_deployment ros_gz_epd_pipeline.launch.py` starts:

1. Gazebo Sim using `ros_gz_sim/gz_sim.launch.py`
2. A spawned in-sim RGBD camera model (`ros_gz_sim create -string ...`)
3. `ros_gz_bridge parameter_bridge` for:
   - `/epd/camera/image`
   - `/epd/camera/depth_image`
   - `/epd/camera/camera_info`
4. EPD (`run.launch.py`) remapped to those topics

## Prerequisites

- `ros_gz_sim` and `ros_gz_bridge` installed in your ROS 2 environment.
- `easy_perception_deployment` and `epd_msgs` built and sourced.

## Run

```bash
# in your workspace root
colcon build --packages-up-to easy_perception_deployment
source install/setup.bash

ros2 launch easy_perception_deployment ros_gz_epd_pipeline.launch.py
```

## Minimal validation

In another terminal (with the same workspace sourced), run:

```bash
bash $(ros2 pkg prefix easy_perception_deployment)/share/easy_perception_deployment/launch/wait_for_epd_output.sh
```

By default this checks for one message on `/easy_perception_deployment/epd_p2_output`.

Optional overrides:

```bash
EPD_OUTPUT_TOPIC=/easy_perception_deployment/epd_p3_output \
EPD_OUTPUT_TIMEOUT=45 \
bash $(ros2 pkg prefix easy_perception_deployment)/share/easy_perception_deployment/launch/wait_for_epd_output.sh
```

## Notes

- This launch is additive and does **not** modify Gazebo Classic flows.
- The camera model uses an inline SDF `rgbd_camera` sensor with base topic `/epd/camera`.
  Gazebo Sim publishes the corresponding image/depth/camera_info streams, bridged into ROS 2.
