# EPD Known Limitations

This document records current product boundaries for the EPD-0 → EPD-9 line. A limitation is not automatically a defect; it is a capability or evidence boundary that must not be overstated during release/demo acceptance.

## 1. Supported baseline

The maintained reference baseline is Ubuntu 22.04 + ROS 2 Humble.

Other operating systems, ROS distributions and container/runtime combinations may work but are not accepted by this roadmap unless separately validated.

## 2. RealSense reference path

The primary 3D reference camera is an Intel RealSense D435i-style RGB + aligned-depth + CameraInfo pipeline.

EPD can use other ROS image topics, but camera namespace, depth alignment, intrinsics and QoS must be validated on that camera. Camera Assistant does not guarantee arbitrary camera-driver compatibility.

## 3. 3D geometry depends on valid depth

Localization/Tracking geometry can degrade when:

- depth is missing or sparse;
- reflective/transparent surfaces produce invalid depth;
- object masks contain too few valid depth points;
- RGB/depth are misaligned;
- CameraInfo is wrong or stale;
- the object lies outside useful depth range.

EPD-6 exposes this evidence. It does not make invalid depth physically valid.

## 4. Plane/background filtering is not enabled by default

EPD-6 intentionally did not add automatic table-plane/background removal without measured evidence that the workcell needs it.

If table/background contamination is observed, add filtering as a separately measured perception increment rather than silently changing geometry semantics.

## 5. Confidence on 3D contract output

Current native Localization/Tracking messages do not expose a trustworthy one-to-one per-object confidence association suitable for the EPD-7 normalized Workcell contract.

EPD-7 therefore does not fabricate confidence. Downstream consumers must treat missing confidence as unavailable, not as zero or one.

## 6. Localization IDs are not persistent Tracking IDs

Tracking mode preserves the actual EPD stable IDs.

Localization-only contract identities are observation-scoped. They must not be interpreted as persistent object identity across frames.

## 7. Collision geometry is conservative

The normalized contract only publishes positive observed dimensions when available and currently represents supported observed collision geometry as a box.

EPD does not invent dimensions when geometry is missing. Downstream PlanningScene code remains responsible for validating geometry before applying it.

## 8. Live preview is an operator aid

The EPD-2 live preview helps human validation. It is not a safety-rated vision display and must not be used as the only evidence for robot-motion safety.

Preview FPS may differ from inference/output rate depending on transport, visualization and GUI rendering overhead.

## 9. Topic discovery can time out

ROS graph discovery is asynchronous and can be slow on unhealthy or busy graphs.

EPD preserves configured camera topics when discovery fails, but a configured topic is not equivalent to a live sampled stream. Camera Assistant is the stronger health check.

## 10. Model introspection has limits

ONNX metadata does not always reveal enough semantic information to prove:

- exact training dataset;
- class-name ordering;
- post-processing expectations;
- preprocessing beyond graph-visible operations;
- whether an arbitrary exported model follows EPD's expected detector output contract.

The Smart Model Manager reports what can be verified and should not present inferred properties as certain.

## 11. Training Studio is not a general-purpose training platform

EPD-4 improves observability/recovery around the existing training pipeline. It does not replace a full experiment-management system.

Training loss alone is insufficient evidence of model quality. Validation metrics and real workcell testing remain necessary.

## 12. EPD profiles are configuration provenance, not certification

EPD-5 profiles capture reproducible settings and asset hashes where available.

A profile becomes "known-good" only because an operator has accepted that exact configuration on a target environment. The flag itself is not proof of correctness.

## 13. Rosbag replay requires topic compatibility

EPD does not silently rewrite arbitrary recorded topic names. The active profile and recorded rosbag topics must be compatible or explicitly handled by the operator.

## 14. Deterministic replay is not a full live-camera test

Deterministic replay is excellent regression evidence for inference/tracking semantics and backpressure, but it does not test:

- USB/camera hardware reliability;
- live exposure/lighting variation;
- RealSense driver stability;
- real network latency;
- physical scene changes outside fixture coverage.

A target release should include both replay evidence and a live-camera smoke test when camera hardware is part of the product claim.

## 15. CUDA availability is build + runtime dependent

Selecting CUDA requires:

- an ONNX Runtime build containing CUDA support;
- compatible NVIDIA driver/runtime;
- accessible GPU device;
- compatible deployment image/native environment.

Explicit CUDA is designed to fail rather than silently claim GPU execution when these conditions are not met.

## 16. TensorRT is deliberately gated

The current upstream `epd_onnxruntime_vendor` CUDA path does not itself enable TensorRT.

EPD-8 contains conditional EPD-side TensorRT integration, but TensorRT remains unavailable until the vendor/runtime is built with TensorRT support and a compatible image/environment is provided.

This is an intentional truth boundary, not a promise that TensorRT works out of the box.

## 17. Jetson requires target-compatible build/runtime

Jetson is an NVIDIA aarch64 platform and should use a JetPack/L4T-compatible environment or validated native build.

An x86 CUDA image must not be assumed compatible with Jetson.

## 18. Performance benchmarks are workload-specific

CPU/CUDA/TensorRT performance depends on:

- model;
- input size;
- object count;
- hardware;
- provider version;
- threading;
- visualization/mask settings;
- thermal/power state.

Do not generalize one benchmark result to another machine/model without re-measuring.

## 19. Workcell contract is perception-only

EPD-7 does not own:

- scene authoring;
- task binding;
- PlanningScene writes;
- grasp selection;
- MoveIt planning;
- execution approval;
- robot motion.

Downstream Workcell Studio / EMD safety and motion gates remain authoritative.

## 20. No real robot motion is required for EPD acceptance

EPD release/demo acceptance should be completed without commanding robot motion.

Any later real-hardware execution must follow the separate guarded hardware workflow and must not weaken collision checking or other safety gates to make a demo pass.

## 21. Diagnostics bundles are best-effort evidence

`epd_diagnostics_bundle.py` intentionally continues when optional commands/topics are unavailable. The manifest records missing/time-limited probes.

By default HOME/user strings are redacted, but a bundle can still contain model names, topic names, labels and captured runtime output. Review it before sharing outside the project.

## 22. Screenshot evidence can contain sensitive content

Camera images, paths, usernames, customer labels and workcell layouts can appear in screenshots.

Review/redact release screenshots before external sharing.

## 23. Release acceptance is not a safety certificate

`epd_release_acceptance.py`, replay PASS, diagnostics PASS, healthy camera status and Workcell contract READY are engineering acceptance evidence only.

They do not certify functional safety, machinery compliance or safe autonomous robot motion.
