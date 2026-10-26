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

```bash
ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true
EPD_EXECUTION_BACKEND=cpu ros2 launch easy_perception_deployment epd_emd_pipeline.launch.py \
  camera_source:=realsense_unaligned usecase_mode_override:=3 \
  publish_workcell_contract:=true contract_source_mode:=localization \
  contract_require_tracking_ids:=false workcell_scene_id:=<scene id>
```
