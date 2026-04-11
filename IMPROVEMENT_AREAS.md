# Improvement Areas Audit

_Last reviewed: 2026-04-11 (UTC)_

This document highlights concrete areas in the repository that likely need attention.
Each item includes evidence and a recommended next step.

## 1) Source/layout hygiene in CMake target composition (High)

**What to improve**
- C++ implementation files are stored under `include/` and then compiled from there.

**Evidence**
- `easy_perception_deployment/easy_perception_deployment/CMakeLists.txt` compiles:
  - `include/epd_utils_lib/epd_container.cpp`
  - `include/ort_cpp_lib/ort_base.cpp`
  - `include/ort_cpp_lib/p3_ort_base.cpp`
  - `include/ort_cpp_lib/p2_ort_base.cpp`

**Why it matters**
- Keeping implementation files in `include/` makes dependency boundaries harder to maintain and can confuse IDE indexing, packaging rules, and contributors expecting a standard `src/` layout.

**Recommended action**
- Move `.cpp` files into `src/` and keep headers in `include/`.
- Update `target_sources()`/`set(...)` paths accordingly.
- Add a short note in CONTRIBUTING about code layout conventions.

## 2) Configure-time model download policy and reproducibility gap (High)

**What to improve**
- Model download flow is robust overall, but one model (`ssd_mobilenet_v1_12.onnx`) has no SHA256 verification configured.

**Evidence**
- `EPD_MODEL_SSD_MOBILENET_SHA256` is set to an empty string in CMake.

**Why it matters**
- Missing hash checks reduce supply-chain confidence and reproducibility.

**Recommended action**
- Add and enforce the SHA256 for SSD Mobilenet.
- Consider flipping `EPD_REQUIRE_MODELS` default to `OFF` for CI/buildfarm and gate model-dependent tests with explicit labels.

## 3) CLI config script: error handling and maintainability (High)

**What to improve**
- `config_epd.py` has extensive `sys.exit(...)` paths inside helper methods and still contains an unresolved TODO for CLI options.

**Evidence**
- TODO note in `parse_args` mentions missing options.
- Several helper functions (`_load_json_config`, `normalize_color_histogram_metric`, validators) terminate directly instead of raising typed exceptions.

**Why it matters**
- Direct process exits make the module harder to reuse as a library and complicate unit testing.

**Recommended action**
- Replace internal `sys.exit(...)` with custom exceptions and keep process exit at `main()` only.
- Migrate from `getopt` to `argparse` fully (already imported), including type validation and better help UX.

## 4) GUI test suite uses shell subprocesses for filesystem operations (Medium)

**What to improve**
- Tests invoke `rm` via `subprocess.Popen` to delete config files.

**Evidence**
- `easy_perception_deployment/easy_perception_deployment/gui/test_gui.py` uses `subprocess.Popen(['rm', ...])` for cleanup.

**Why it matters**
- This is less portable and less readable than native Python filesystem APIs; it also introduces avoidable process overhead.

**Recommended action**
- Replace subprocess cleanup with `Path.unlink(missing_ok=True)` or `os.remove` with explicit checks.
- Prefer test fixtures for setup/teardown to isolate state.

## 5) Dependency pinning strategy in GUI requirements (Medium)

**What to improve**
- `requirements.txt` mixes exact pins, loose minimums, and unpinned packages for heavyweight dependencies.

**Evidence**
- `PySide6` is unpinned.
- `pycocotools>=2.0.6` and `docker>=6.0.0` are lower-bound only.
- `torch==2.7.1` and `torchvision==0.22.0` are pinned.

**Why it matters**
- Mixed strategies can create non-reproducible installs and breakages across OS / Python versions.

**Recommended action**
- Adopt a two-file approach:
  - `requirements.in` for human-maintained constraints.
  - compiled lock file (e.g., via `pip-tools`) for reproducible installs.
- Separate CPU/GPU extras for Torch where relevant.

## 6) Minor style and naming cleanup opportunities (Low)

**What to improve**
- Class and variable naming are inconsistent with common Python conventions.

**Evidence**
- Class declaration `class EPDConfigurator():` has unnecessary parentheses.
- Temporary variables named `dict` shadow the built-in `dict` type in multiple locations.

**Why it matters**
- Improves readability, tooling diagnostics, and onboarding ease.

**Recommended action**
- Run a light refactor pass (or apply ruff/flake8 rules) for naming and built-in shadowing.

## Suggested execution order

1. CLI exception model + argparse migration.
2. Add SSD model hash and tighten model reproducibility policy.
3. Move `.cpp` implementations from `include/` to `src/`.
4. Refactor GUI tests to fixture-based file handling.
5. Dependency management lock strategy.
