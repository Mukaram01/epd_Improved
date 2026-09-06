import pytest

from cli.config_schema import (
    ConfigSchemaError,
    normalize_execution_backend,
    validate_session_config,
)


def _session(**updates):
    config = {
        "path_to_model": "model.onnx",
        "path_to_label_list": "labels.txt",
        "visualizeFlag": "robot",
        "useCPU": "CPU",
    }
    config.update(updates)
    return config


def test_legacy_cpu_migrates_to_cpu_backend():
    normalized = validate_session_config(_session())
    assert normalized["execution_backend"] == "cpu"


def test_legacy_gpu_migrates_to_cuda_backend():
    normalized = validate_session_config(_session(useCPU="GPU"))
    assert normalized["execution_backend"] == "cuda"


def test_explicit_auto_is_preserved():
    normalized = validate_session_config(_session(execution_backend="auto"))
    assert normalized["execution_backend"] == "auto"


def test_backend_aliases_are_normalized():
    assert normalize_execution_backend("GPU") == "cuda"
    assert normalize_execution_backend("trt") == "tensorrt"


def test_unknown_backend_is_rejected():
    with pytest.raises(ConfigSchemaError):
        validate_session_config(_session(execution_backend="openvino"))


def test_gpu_index_must_be_non_negative():
    with pytest.raises(ConfigSchemaError):
        validate_session_config(
            _session(execution_backend="cuda", execution_backend_gpu_index=-1)
        )
