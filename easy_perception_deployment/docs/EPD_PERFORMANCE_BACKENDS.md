# EPD-8 — Performance Backends

EPD-8 makes inference execution explicit and measurable. It adds CPU/CUDA/TensorRT backend selection, host/runtime probing, a deterministic backend benchmark, and a guarded Jetson path while keeping CPU as the reliable baseline.

This increment does **not** change the perception model, ROS message schemas, Workcell Studio scene ownership, grasp planning, MoveIt, or robot motion.

## Operator workflow

Open **Deploy → Performance** or press `Ctrl+Shift+B`.

The Performance Backends window shows measured evidence for:

- host architecture;
- NVIDIA Jetson markers;
- Docker availability;
- NVIDIA host/runtime availability;
- CPU/GPU/TensorRT image presence;
- compiled EPD provider capability through `epd_backend_probe`, when the current workspace has been rebuilt;
- the currently selected backend;
- deterministic CPU-versus-accelerated replay benchmarking.

The selected backend is stored in `config/session_config.json`:

```json
{
  "useCPU": "CPU",
  "execution_backend": "auto",
  "execution_backend_gpu_index": 0
}
```

`useCPU` is retained for compatibility with older EPD profiles and scripts. `execution_backend` is the EPD-8 source of truth.

## Backend semantics

### AUTO

`auto` is the recommended starting point.

For Docker Deploy, the launcher first checks whether the host has an NVIDIA execution path. It uses the GPU image only when NVIDIA runtime evidence exists. If the GPU image is not present, AUTO falls back to the CPU image.

Inside ONNX Runtime, AUTO attempts CUDA only in a GPU-capable build. If CUDA provider initialization fails, it reports the failure and creates a CPU session instead.

AUTO does **not** select TensorRT implicitly. TensorRT can have model-specific engine-build behavior, workspace/memory requirements and unsupported partitions, so it must be selected and benchmarked explicitly.

### CPU

CPU is the reliable baseline. No CUDA or TensorRT provider is appended to the ONNX Runtime session.

Use CPU when:

- debugging model/camera correctness;
- validating a new model;
- an NVIDIA runtime is unavailable;
- comparing accelerated semantics against a known baseline;
- maximum portability matters more than throughput.

### CUDA

CUDA is an explicit NVIDIA ONNX Runtime execution provider.

If CUDA is selected and provider initialization fails, EPD fails clearly. It does not silently claim a GPU run while executing on CPU.

Typical native accelerated build:

```bash
cd ~/epd_ros2_ws/src/easy_perception_deployment/easy_perception_deployment
bash scripts/build_accelerated_backend.sh cuda
source ~/epd_ros2_ws/install/setup.bash
ros2 run easy_perception_deployment epd_backend_probe
```

Expected capability probe for a CUDA build:

```json
{"schema_version":"epd_backend_probe/v1","architecture":"x86_64","cpu":true,"cuda":true,"tensorrt":false}
```

The exact architecture can differ.

### TensorRT

TensorRT is supported as an **opt-in provider build**, not as an assumed capability of the standard GPU image.

Three conditions must all be true:

1. `epd_onnxruntime_vendor` must build ONNX Runtime with TensorRT.
2. EPD must be configured with `-DEPD_ENABLE_TENSORRT=ON`.
3. Docker Deploy must be given an explicit TensorRT-capable image through `EPD_TENSORRT_IMAGE`.

The upstream `ros-industrial/epd_onnxruntime_vendor` used by the original EPD project enables CUDA when `USE_CUDA` is set, but does not currently add the TensorRT CMake arguments. A TensorRT-capable vendor fork/overlay therefore needs to extend its `extra_cmake_args` with the equivalent of:

```cmake
if(DEFINED USE_TENSORRT)
  list(APPEND extra_cmake_args "-Donnxruntime_USE_TENSORRT=ON")
  list(APPEND extra_cmake_args "-Donnxruntime_TENSORRT_HOME=${TENSORRT_HOME}")
endif()
```

The vendor must also keep CUDA enabled because EPD appends CUDA behind TensorRT as the fallback for graph partitions TensorRT does not execute.

After the vendor is TensorRT-capable:

```bash
export TENSORRT_HOME=/path/to/tensorrt
bash scripts/build_accelerated_backend.sh tensorrt
source ~/epd_ros2_ws/install/setup.bash
ros2 run easy_perception_deployment epd_backend_probe
```

A TensorRT-ready probe must report:

```json
{"cpu":true,"cuda":true,"tensorrt":true}
```

For GUI/Docker deployment also set an image that contains the matching CUDA, TensorRT, cuDNN and ONNX Runtime stack:

```bash
export EPD_TENSORRT_IMAGE=my-org/epd-humble:tensorrt
```

EPD intentionally refuses TensorRT Docker deployment when this image is not explicitly supplied. It will not reuse the ordinary GPU image and pretend TensorRT is active.

## Docker image overrides

The Deploy wrapper supports:

```bash
export EPD_CPU_IMAGE=cardboardcode/epd-humble-base:CPU
export EPD_GPU_IMAGE=cardboardcode/epd-humble-base:GPU
export EPD_TENSORRT_IMAGE=my-org/epd-humble:tensorrt
```

`EPD_CPU_IMAGE` and `EPD_GPU_IMAGE` have legacy defaults. `EPD_TENSORRT_IMAGE` has **no default** by design.

The wrapper passes the selected backend into the container as:

```text
EPD_EXECUTION_BACKEND=cpu|cuda|tensorrt
EPD_GPU_INDEX=<non-negative integer>
```

The same variables can be used for a native launch.

## Jetson path

Jetson is treated as a distinct NVIDIA `aarch64` target rather than assuming an x86 CUDA image will work.

The Performance Backends probe checks `/etc/nv_tegra_release`, architecture, and the device-tree model. The build helper provides a guarded native CUDA path:

```bash
bash scripts/build_accelerated_backend.sh jetson
```

It requires:

- `aarch64`;
- a Jetson/NVIDIA platform marker;
- JetPack CUDA/cuDNN installed;
- `epd_onnxruntime_vendor` present in the workspace.

For containerized Jetson deployment, provide an image built for the installed JetPack/L4T release:

```bash
export EPD_GPU_IMAGE=my-org/epd-humble:jetson-jp6
```

The launcher uses the NVIDIA runtime on Jetson. EPD does not automatically pull or claim compatibility for an x86 GPU image.

A native Jetson build can also be run through the normal ROS launch files without the GUI Docker wrapper after sourcing the workspace.

## Compiled capability probe

After rebuilding, run:

```bash
ros2 run easy_perception_deployment epd_backend_probe
```

This reports **build capability only**. It does not prove that a physical GPU, driver, container runtime or TensorRT engine is healthy. The GUI combines this with host/runtime evidence.

## Deterministic backend benchmark

EPD-8 benchmarks execution providers with the existing P8 deterministic tracking replay rather than a synthetic matrix-multiply benchmark.

Example:

```bash
ros2 run easy_perception_deployment epd_backend_benchmark.py \
  --backends cpu,cuda \
  --fixture ~/epd_ros2_ws/src/easy_perception_deployment/easy_perception_deployment/fixtures/p8_tracking.json \
  --output /tmp/epd_backend_benchmark.json
```

For TensorRT after it is provisioned:

```bash
ros2 run easy_perception_deployment epd_backend_benchmark.py \
  --backends cpu,cuda,tensorrt \
  --fixture ~/epd_ros2_ws/src/easy_perception_deployment/easy_perception_deployment/fixtures/p8_tracking.json
```

Each backend launches the existing production replay path with an explicit `EPD_EXECUTION_BACKEND`. The report includes:

- replay PASS/FAIL;
- wall-clock run time;
- minimum/average/maximum inference latency from EPD diagnostics;
- inference rate;
- observation rate;
- stable Tracking IDs;
- LOST IDs;
- geometry-quality summary;
- whether the accelerated semantic summary exactly matches the CPU baseline.

A speed improvement is not sufficient for acceptance. The accelerated run must first PASS the existing replay acceptance checks. Any semantic mismatch with CPU is called out for review.

## Replay performance fields

The deterministic replay summary now includes:

```json
{
  "performance": {
    "execution_backend": "cuda",
    "inference_latency_min_ms": 0.0,
    "inference_latency_avg_ms": 0.0,
    "inference_latency_max_ms": 0.0,
    "inference_rate_hz": 0.0,
    "observation_rate_hz": 0.0
  }
}
```

Values come from the production inference diagnostics. They are not invented by the GUI.

## Recommended adoption sequence

For a new workstation or model:

1. Run CPU first and confirm camera/model/replay correctness.
2. Probe CUDA readiness.
3. Run CPU vs CUDA deterministic benchmark.
4. Accept CUDA only when replay passes and semantics match CPU.
5. If CUDA performance is still insufficient, provision TensorRT.
6. Benchmark TensorRT against the same CPU baseline.
7. Test the selected backend with the live RealSense workcell.
8. Save the known-good backend inside an EPD-5 perception profile.

For Jetson, do the same sequence on the actual Jetson device rather than assuming desktop benchmark results transfer.

## Failure behavior

EPD-8 prefers explicit failure over false acceleration claims:

- explicit CUDA + unavailable CUDA provider → deployment failure;
- explicit TensorRT + non-TensorRT build → deployment failure;
- TensorRT Docker selection without `EPD_TENSORRT_IMAGE` → deployment failure;
- invalid GPU index → deployment failure;
- AUTO + unavailable CUDA → CPU fallback;
- missing GPU image under AUTO → CPU image fallback when available.

This makes it possible to know whether acceleration is really in use.

## Ownership and safety

Performance backend selection changes only where ONNX inference executes.

It does not:

- authorize robot motion;
- bypass Workcell Studio readiness;
- alter collision checking;
- write PlanningScene objects;
- change grasp candidates;
- change EPD-7 normalized contract ownership;
- make TensorRT/Jetson hardware automatically safe for deployment.

Real robot motion remains governed by the existing downstream Workcell Studio / EMD safety gates.
