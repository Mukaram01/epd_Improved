# Copyright 2026 Easy Perception Deployment contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
import sys

import pytest


_LAUNCH_DIR = Path(__file__).resolve().parent
if str(_LAUNCH_DIR) not in sys.path:
    sys.path.insert(0, str(_LAUNCH_DIR))

from backend_launch_config import resolve_backend_launch_defaults  # noqa: E402


def _write_config(tmp_path, **overrides):
    payload = {
        "useCPU": "CPU",
        "execution_backend": "auto",
        "execution_backend_gpu_index": 0,
    }
    payload.update(overrides)
    path = Path(tmp_path) / "session_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_saved_cpu_selection_is_honoured(tmp_path):
    path = _write_config(
        tmp_path,
        useCPU="CPU",
        execution_backend="cpu",
    )

    backend, gpu_index = resolve_backend_launch_defaults(path, {})

    assert backend == "cpu"
    assert gpu_index == "0"


def test_saved_cuda_selection_and_gpu_index_are_honoured(tmp_path):
    path = _write_config(
        tmp_path,
        useCPU="GPU",
        execution_backend="cuda",
        execution_backend_gpu_index=2,
    )

    backend, gpu_index = resolve_backend_launch_defaults(path, {})

    assert backend == "cuda"
    assert gpu_index == "2"


def test_environment_override_has_priority_over_saved_config(tmp_path):
    path = _write_config(
        tmp_path,
        execution_backend="cuda",
        execution_backend_gpu_index=3,
    )

    backend, gpu_index = resolve_backend_launch_defaults(
        path,
        {
            "EPD_EXECUTION_BACKEND": "cpu",
            "EPD_GPU_INDEX": "7",
        },
    )

    assert backend == "cpu"
    assert gpu_index == "7"


def test_legacy_usecpu_is_used_when_execution_backend_is_missing(tmp_path):
    path = Path(tmp_path) / "session_config.json"
    path.write_text(
        json.dumps({"useCPU": "CPU"}),
        encoding="utf-8",
    )

    backend, gpu_index = resolve_backend_launch_defaults(path, {})

    assert backend == "cpu"
    assert gpu_index == "0"


def test_legacy_gpu_selection_maps_to_cuda(tmp_path):
    path = Path(tmp_path) / "session_config.json"
    path.write_text(
        json.dumps({"useCPU": "GPU"}),
        encoding="utf-8",
    )

    backend, _ = resolve_backend_launch_defaults(path, {})

    assert backend == "cuda"


def test_invalid_gpu_index_is_rejected(tmp_path):
    path = _write_config(
        tmp_path,
        execution_backend="cpu",
        execution_backend_gpu_index=-1,
    )

    with pytest.raises(RuntimeError, match="non-negative integer"):
        resolve_backend_launch_defaults(path, {})
