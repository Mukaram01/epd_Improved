# Real D435i integration validation — 16 October 2026

Integration PASS; hybrid remains default-OFF and unqualified. Stage-A2 remains **BLOCKED 0/2**.

Merged PR #232 (`62fb324`) into PR #231 (`208a44d`) on a separate integration branch. Localization and tracking use the type-aware mask serializer and exact source cloud header in both modes. Legacy inference/geometry sources and ROS/EMD schemas are unchanged.

Captured 12 native RGB-D frames: four separated (A), four closer/contact (B), four partially overlapping (C). All have exact RGB/depth/CameraInfo stamps, matching 1280×720 calibration and color optical frames. SDK USB descriptor 3.2; depth scale 0.001 m/unit. Actual model: MaskRCNN-10.onnx; COCO list indices banana=47, apple=48, mouse=65. Source and binary seals, every-frame geometry/confidence/ownership/tracking/resource measurements and limitations are in `results.json` and `frames.jsonl`.

| Scene | Legacy geometry-valid | Hybrid geometry-valid | Legacy shared mask pixels | Hybrid shared mask pixels |
|---|---:|---:|---:|---:|
| Separated | 16 | 16 | 0 | 0 |
| Closer/contact | 11 | 5 | 16,841 | 0 |
| Partial overlap | 13 | 7 | 50,100 | 0 |

Hybrid rejects eight proposals for ownership and four for ambiguous axes. All accepted mask encodings/strides/payloads and mask/cloud source headers pass. Default, explicit-disabled and original PR #232 full object messages are byte-identical on all four A frames, including mask/cloud data and poses.

Banana and apple are detected in all 12 frames. Mouse is detected in 3/4 separated frames (0.684–0.799), absent above threshold in B/C. A also contains a second banana-shaped item; false bicycle/bottle and duplicate class proposals remain explicit. Geometry plausibility does not establish semantic correctness. Annotation-dependent precision/recall/IoU/ownership/pose errors are **UNKNOWN**. Tracking IDs remain consistent for observed A objects; ground-truth ID accuracy is UNKNOWN.

Mean processing time: legacy 3,987 ms; hybrid 3,698 ms. Localization peak RSS: 1,485/1,386 MiB; average measured process CPU during round trips: 1.73/1.57 core equivalents. These sequential captured-frame timings are not live FPS. Live camera acquisition worked; sustained direct-camera EPD throughput was not qualified.

Validation: 96 C++ cases across seven focused targets, 56 Python cases, five ROS smoke modes, and all 12 native-frame localization/tracking comparisons pass. The new transport assertion fails on the old PR #231 binary as expected and passes after integration. Initial Cyclone native-frame delivery and an omitted legacy tracker setting failed before successful Fast DDS/KCF runs; logs are preserved. Input republications are recorded separately from selected unique frames.

## Reproduce locally

The sealed full data and executed drivers remain outside Git at `/home/user/epd-v2-artifacts/rgbd-integration-20261016` (A/B/C original CDR messages, RGB/depth arrays, output CDRs, logs, probes and drivers). `input_seals.json` seals inputs and drivers. No frozen Stage-A2 selection payload was opened.

From this integration checkout:

```bash
source /opt/ros/humble/setup.bash
source /home/user/epd_ros2_ws/install/setup.bash
export EPD_EXECUTION_BACKEND=cpu OPENBLAS_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen
export ROS_DOMAIN_ID=173 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
EPD_TEST_ROOT=/home/user/epd-v2-artifacts/rgbd-integration-20261016
cmake -S easy_perception_deployment/easy_perception_deployment -B "$EPD_TEST_ROOT/build" -DBUILD_TESTING=ON
cmake --build "$EPD_TEST_ROOT/build" -j2 --target easy_perception_deployment epd_test_ros_message_transport epd_test_hybrid_rgbd epd_test_geometry_quality epd_test_observation epd_test_temporal_tracker epd_test_time_utils epd_test_inference_scheduler
ctest --test-dir "$EPD_TEST_ROOT/build" --output-on-failure -R '^epd_test_(ros_message_transport|hybrid_rgbd|geometry_quality|observation|temporal_tracker|time_utils|inference_scheduler)$'
python3 -m pytest -q easy_perception_deployment/easy_perception_deployment/scripts/test_workcell_contract_bridge.py easy_perception_deployment/easy_perception_deployment/scripts/test_config_epd.py easy_perception_deployment/easy_perception_deployment/launch/test_backend_launch_config.py easy_perception_deployment/easy_perception_deployment/gui/test_model_output_contract.py
# Python configuration tests modify source fixture configs; restore only these test effects before hardware replay.
git restore --worktree easy_perception_deployment/easy_perception_deployment/config
python3 "$EPD_TEST_ROOT/runtime_test.py" --folders A --modes default disabled hybrid legacy_tracking hybrid_tracking --tag reproduce
python3 "$EPD_TEST_ROOT/runtime_test.py" --folders B C --modes default hybrid legacy_tracking hybrid_tracking --tag reproduce
python3 easy_perception_deployment/easy_perception_deployment/test/hybrid_rgbd_runtime_smoke.py --binary "$EPD_TEST_ROOT/build/easy_perception_deployment" --image easy_perception_deployment/easy_perception_deployment/test/nadezhda_diskant.jpg --output /tmp/epd-integration-new-smoke
```

Use the existing local standard ONNX assets; no download/training is required. Existing build/model symlinks were used. Hardware replay drivers pin this checkout/package path and evidence location; preserve the recorded originals when choosing a new output tag. `build_pr232_baseline.py` builds the original PR #232 main header against verified byte-identical shared implementation objects. `detector_probe.cpp` measures actual production C++ detector confidences and pre-refinement ownership with the same CHW tensor shape as EPD.

No PR merge, EMD change, robot motion, model training, threshold weakening or Stage-A2 promotion.
