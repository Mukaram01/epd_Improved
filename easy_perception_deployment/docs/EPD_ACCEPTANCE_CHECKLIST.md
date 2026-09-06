# EPD Release Acceptance Checklist

Use this checklist on the actual target workstation. Record PASS / WARN / FAIL and attach evidence. Do not convert missing evidence into a PASS.

## A. Build / environment

- [ ] Ubuntu 22.04 target identified.
- [ ] ROS 2 Humble sourced.
- [ ] Workspace builds successfully.
- [ ] EPD package is sourced from the intended workspace/install.
- [ ] Exact EPD commit SHA recorded.
- [ ] Model and label assets are present.
- [ ] Selected execution backend is explicit and understood.

## B. Launcher / Help

- [ ] Launcher opens without exception.
- [ ] Train opens and closes normally.
- [ ] Deploy opens and closes normally.
- [ ] Help & Guides opens from launcher.
- [ ] F1 opens Help from Launcher, Train and Deploy.

## C. Camera truth

- [ ] Saved RGB topic remains visible if discovery times out.
- [ ] Configured / Detected / Missing state is truthful.
- [ ] Camera Assistant opens.
- [ ] RGB topic is sampled live.
- [ ] For Localization/Tracking: aligned depth is sampled live.
- [ ] For Localization/Tracking: CameraInfo is sampled live.
- [ ] Resolution and encoding are plausible.
- [ ] Stopped/unresponsive streams are not labelled live.

Reference RealSense topics:

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/camera_info
```

## D. Model truth

- [ ] ONNX model path exists.
- [ ] Model inspection succeeds.
- [ ] Label list exists.
- [ ] Label/model compatibility has evidence where verifiable.
- [ ] Selected use case is compatible with the model.
- [ ] No unverifiable model property is presented as confirmed truth.

## E. Live operator loop

- [ ] Deploy can start perception.
- [ ] Live Perception View updates.
- [ ] Detection overlay visibly works when enabled.
- [ ] Turning overlay off does not stop ROS perception results.
- [ ] Object count is derived from EPD output.
- [ ] FPS/latency are shown only when available.
- [ ] Stale preview/result leaves LIVE state.
- [ ] Deploy stops perception cleanly.

## F. 3D / Tracking

For the manipulation-facing reference demo:

- [ ] Tracking mode selected.
- [ ] RGB/depth/CameraInfo health passes.
- [ ] 3D Inspector opens.
- [ ] Result/depth dimensions agree.
- [ ] Depth encoding supported.
- [ ] Intrinsics finite and positive.
- [ ] Source/result timestamps align.
- [ ] Object centroid is finite where geometry is valid.
- [ ] Positive observed dimensions are reported where available.
- [ ] Stable Tracking IDs persist across updates.
- [ ] LOST IDs are surfaced when tracks expire.
- [ ] Invalid/missing geometry is not replaced by guessed collision geometry.

## G. Profiles / replay

- [ ] Current deploy state can be saved as an EPD profile.
- [ ] Model SHA256 captured when model exists.
- [ ] Label SHA256 captured when labels exist.
- [ ] Profile restore is blocked while perception is active.
- [ ] Profile restore reproduces model/labels/topic/mode/runtime settings.
- [ ] Backend and GPU index are restored with EPD-8 profiles.
- [ ] Deterministic replay completes.
- [ ] Replay summary result is PASS.
- [ ] Replay stable-ID/lifecycle evidence is present.
- [ ] Replay stale-result count is acceptable.

## H. Workcell Studio / EMD contract

When Workcell integration is part of acceptance:

- [ ] Contract publishing is explicitly enabled.
- [ ] `scene_id` supplied by Workcell/test launch is correct.
- [ ] `camera_id` supplied by Workcell/test launch is correct.
- [ ] Optional `profile_ref` is provenance only.
- [ ] Snapshot schema is `workcell_perception_snapshot/v1`.
- [ ] Status schema is `workcell_perception_status/v1`.
- [ ] Tracking IDs match native EPD IDs.
- [ ] Lost IDs match native EPD lost IDs.
- [ ] Timestamp/frame are preserved from native perception truth.
- [ ] Missing dimensions/confidence are not fabricated.
- [ ] EPD performs no PlanningScene write, grasp selection, MoveIt call or robot motion.

## I. Performance backend

CPU is always the baseline/recovery path.

- [ ] `epd_backend_probe` run on target machine.
- [ ] Requested backend capability is actually compiled/available.
- [ ] Explicit CUDA does not silently run CPU when CUDA is unavailable.
- [ ] Explicit TensorRT remains blocked unless the vendor/runtime/image support it.
- [ ] Jetson path uses target-compatible native/L4T environment.
- [ ] CPU deterministic replay PASS recorded.
- [ ] Accelerated deterministic replay PASS recorded, if used.
- [ ] Accelerated semantic result compared against CPU.
- [ ] Performance improvement measured rather than assumed.

## J. Release evidence

Recommended command:

```bash
ros2 run easy_perception_deployment epd_release_acceptance.py \
  --with-replay \
  --output /tmp/epd_release_acceptance.json
```

Recommended diagnostics bundle:

```bash
ros2 run easy_perception_deployment epd_diagnostics_bundle.py \
  --output /tmp/epd_diagnostics.zip
```

Attach:

- [ ] commit SHA;
- [ ] acceptance JSON;
- [ ] diagnostics ZIP;
- [ ] known-good profile;
- [ ] deterministic replay summary;
- [ ] backend benchmark report if accelerated backend is used;
- [ ] screenshots/demo evidence;
- [ ] explicit notes for WARN items;
- [ ] known-limitations review.

## K. Safety / ownership boundary

- [ ] Acceptance did not require real robot motion.
- [ ] No safety gate was bypassed to make perception pass.
- [ ] Perception health is not treated as a motion-safety certificate.
- [ ] Workcell Studio/EMD retains task/planning/motion ownership.
- [ ] Real hardware motion, if later enabled, follows the separate guarded hardware acceptance process.

## Final release decision

```text
PASS  = required evidence is present and no blocking failure remains.
WARN  = non-blocking limitation is understood, documented and accepted.
FAIL  = blocking requirement failed or evidence is missing for a claimed capability.
```

A release/demo configuration should not be called ready while any required item remains FAIL.
