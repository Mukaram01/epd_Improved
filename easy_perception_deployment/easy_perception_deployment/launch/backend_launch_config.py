"""Resolve EPD execution backend defaults for ROS 2 launch.

The Deploy GUI stores the operator selection in ``config/session_config.json``.
The C++ ONNX Runtime layer consumes ``EPD_EXECUTION_BACKEND`` and
``EPD_GPU_INDEX``.  This helper bridges those two sources so a normal
``ros2 launch easy_perception_deployment run.launch.py`` honours the same
backend selected in Deploy.

Explicit environment variables remain higher priority than the saved profile,
and explicit ROS 2 launch arguments can in turn override these defaults.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


BACKENDS = ("auto", "cpu", "cuda", "tensorrt")
_ALIASES = {
    "gpu": "cuda",
    "nvidia": "cuda",
    "trt": "tensorrt",
    "default": "auto",
}


def normalize_backend(value, legacy_backend="cpu"):
    """Normalize EPD backend names while retaining the legacy CPU/GPU fallback."""
    text = str(value or "").strip().lower()
    if not text:
        text = str(legacy_backend or "cpu").strip().lower()
    text = _ALIASES.get(text, text)
    return text if text in BACKENDS else "auto"


def _read_session_config(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _legacy_backend(config):
    value = str(config.get("useCPU", "CPU") or "CPU").strip().lower()
    return "cpu" if value in ("cpu", "true", "1", "yes", "on") else "cuda"


def _gpu_index(value, source):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{source} must be a non-negative integer, got {value!r}."
        ) from exc
    if parsed < 0:
        raise RuntimeError(
            f"{source} must be a non-negative integer, got {parsed}."
        )
    return str(parsed)


def resolve_backend_launch_defaults(session_config_path, environ=None):
    """Return ``(backend, gpu_index)`` defaults for ``run.launch.py``.

    Precedence is intentionally explicit:

    1. ``EPD_EXECUTION_BACKEND`` / ``EPD_GPU_INDEX`` environment overrides.
    2. ``execution_backend`` / ``execution_backend_gpu_index`` saved by Deploy.
    3. Legacy ``useCPU`` compatibility field.
    4. CPU / GPU index 0 safe defaults.
    """
    env = os.environ if environ is None else environ
    config = _read_session_config(session_config_path)
    legacy_backend = _legacy_backend(config)

    env_backend = str(env.get("EPD_EXECUTION_BACKEND", "") or "").strip()
    config_backend = config.get("execution_backend")
    backend = normalize_backend(
        env_backend if env_backend else config_backend,
        legacy_backend,
    )

    env_gpu_index = str(env.get("EPD_GPU_INDEX", "") or "").strip()
    if env_gpu_index:
        gpu_index = _gpu_index(env_gpu_index, "EPD_GPU_INDEX")
    else:
        gpu_index = _gpu_index(
            config.get("execution_backend_gpu_index", 0),
            "execution_backend_gpu_index",
        )

    return backend, gpu_index
