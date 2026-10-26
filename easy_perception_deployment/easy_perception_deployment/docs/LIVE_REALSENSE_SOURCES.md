# Live RealSense sources

`run.launch.py` and `epd_emd_pipeline.launch.py` take `camera_source`. Both
sources feed the same ingress outputs (`/easy_perception_deployment/ingress/...`),
so the EPD node, latest-only inference, localization/tracking outputs and the
Workcell contract bridge are unchanged.

| `camera_source` | Camera topics | Depth handling |
|---|---|---|
| `realsense_aligned` (default) | `color/image_raw`, `color/camera_info`, `aligned_depth_to_color/image_raw` | Forwarded untouched; needs `align_depth.enable:=true` on the driver. |
| `realsense_unaligned` | `color/image_raw`, `color/camera_info`, `depth/image_rect_raw`, `depth/camera_info` | Registered into the colour optical frame by `epd_sensor_ingress`. |

Explicit `rgb_topic`, `camera_info_topic`, `depth_topic` and
`depth_camera_info_topic` arguments override the preset topics.

## Registration (`depth_alignment: register`)

- Pairing is latest-frame-wins: only the newest colour frame is pending. A colour frame
  is paired with the colour CameraInfo carrying its exact stamp and the nearest depth
  frame within `max_rgbd_pair_skew_s` (default 20 ms). Unpairable frames are dropped
  and counted, never processed late.
- Each depth pixel is deprojected with the depth intrinsics, transformed by the driver's
  static `depth optical -> colour optical` TF, and projected with the colour intrinsics.
  The nearest surface wins, and the written value is the colour-frame range.
- The registered depth takes the colour image's size, `frame_id` and sensor stamp, the
  same convention as the driver's `aligned_depth_to_color`. The measured pair skew is
  reported in `/easy_perception_deployment/sensor_ingress/health`
  (`last_rgbd_pair_skew_ns`, `max_rgbd_pair_skew_ns`).
- The ingress refuses to publish when it can't register honestly: missing depth
  CameraInfo or extrinsics TF, a frame or size mismatch, non-zero distortion, or an
  unsupported encoding. The reason appears in the health message
  (`last_registration_error`).

Poses stay in the colour optical frame. Camera-to-world calibration is outside EPD.

## CPU profile and backend

On CPU, inference cost follows the colour resolution, so lower colour to 640x480. Keep
depth at 848x480, the D435i's recommended depth-accuracy mode. Registration reprojects
through both cameras' own intrinsics, so unequal sizes are expected and lose no accuracy.
At 15 fps, MaskRCNN-10 measured about 430 ms per frame on LENLEG, against about 880 ms
with 1280x720 colour.

`epd_emd_pipeline.launch.py` takes `execution_backend` and
`execution_backend_gpu_index`, the same as `run.launch.py`. They default to the Deploy
selection in `session_config.json`, and `EPD_EXECUTION_BACKEND` / `EPD_GPU_INDEX` still
override that selection. CPU therefore needs no environment variable, and a GPU is
still chosen explicitly, e.g. `execution_backend:=cuda`.

```bash
ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true \
  rgb_camera.color_profile:=640x480x15 depth_module.depth_profile:=848x480x15
ros2 launch easy_perception_deployment epd_emd_pipeline.launch.py \
  camera_source:=realsense_unaligned usecase_mode_override:=3 \
  publish_workcell_contract:=true contract_source_mode:=localization \
  contract_require_tracking_ids:=false workcell_scene_id:=<scene id>
```
