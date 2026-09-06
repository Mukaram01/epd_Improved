import importlib.util
from pathlib import Path

from windows.backend_manager import (
    backend_status,
    normalize_backend,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "scripts" / "epd_backend_benchmark.py"
SPEC = importlib.util.spec_from_file_location("epd_backend_benchmark", BENCHMARK)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def _probe(**updates):
    probe = {
        "recommended": "cpu",
        "ready": {"cpu": True, "cuda": False, "tensorrt": False},
        "images": {
            "cpu": {"name": "cpu", "present": True},
            "cuda": {"name": "gpu", "present": False},
            "tensorrt": {"name": "", "present": False},
        },
        "compiled": {"cpu": True, "cuda": False, "tensorrt": False},
    }
    probe.update(updates)
    return probe


def test_backend_normalization_keeps_legacy_compatibility():
    assert normalize_backend("", True) == "cpu"
    assert normalize_backend("", False) == "cuda"
    assert normalize_backend("GPU", True) == "cuda"
    assert normalize_backend("trt", True) == "tensorrt"
    assert normalize_backend("unknown", True) == "auto"


def test_auto_reports_current_measured_resolution():
    state, detail = backend_status(_probe(), "auto")
    assert state == "READY"
    assert "CPU" in detail


def test_tensorrt_is_blocked_without_explicit_image():
    state, detail = backend_status(_probe(), "tensorrt")
    assert state == "BLOCKED"
    assert "EPD_TENSORRT_IMAGE" in detail


def test_cuda_ready_only_when_probe_says_ready():
    probe = _probe(
        recommended="cuda",
        ready={"cpu": True, "cuda": True, "tensorrt": False},
    )
    state, _ = backend_status(probe, "cuda")
    assert state == "READY"


def test_benchmark_backend_list_keeps_cpu_first():
    assert BENCH.parse_backend_list("cuda,cpu,tensorrt") == [
        "cpu", "cuda", "tensorrt"
    ]


def test_benchmark_rejects_unknown_backend():
    try:
        BENCH.parse_backend_list("cpu,openvino")
    except ValueError as exc:
        assert "Unsupported backend" in str(exc)
    else:
        raise AssertionError("unknown backend should be rejected")


def test_cpu_semantic_equivalence_is_explicit():
    signature = {
        "result": "PASS",
        "completed_result_count": 2,
        "stable_ids": ["1"],
        "lost_track_ids": [],
        "geometry_quality": {"valid": 1},
        "stale_result_count": 0,
    }
    records = [
        {"backend": "cpu", "status": "PASS", "semantic_signature": signature},
        {"backend": "cuda", "status": "PASS", "semantic_signature": dict(signature)},
    ]
    BENCH.compare_with_cpu(records)
    assert records[0]["equivalent_to_cpu"] is True
    assert records[1]["equivalent_to_cpu"] is True


def test_semantic_mismatch_is_not_hidden():
    cpu = {
        "result": "PASS",
        "completed_result_count": 2,
        "stable_ids": ["1"],
        "lost_track_ids": [],
        "geometry_quality": {"valid": 1},
        "stale_result_count": 0,
    }
    cuda = dict(cpu)
    cuda["stable_ids"] = ["2"]
    records = [
        {"backend": "cpu", "status": "PASS", "semantic_signature": cpu},
        {"backend": "cuda", "status": "PASS", "semantic_signature": cuda},
    ]
    BENCH.compare_with_cpu(records)
    assert records[1]["equivalent_to_cpu"] is False
    assert records[1]["warnings"]
