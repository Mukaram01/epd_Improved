#!/usr/bin/env python3
"""Benchmark EPD execution backends against the deterministic replay fixture.

The benchmark never treats speed as correctness. Each run must first satisfy the
existing replay acceptance checks; accelerated results are then compared with the
CPU baseline for stable-ID/lifecycle and geometry-summary consistency.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

BACKENDS = ("cpu", "cuda", "tensorrt")


def parse_backend_list(value):
    items = []
    for raw in str(value or "").split(","):
        backend = raw.strip().lower()
        if not backend:
            continue
        aliases = {"gpu": "cuda", "trt": "tensorrt"}
        backend = aliases.get(backend, backend)
        if backend not in BACKENDS:
            raise ValueError(
                f"Unsupported backend {backend!r}; choose cpu,cuda,tensorrt"
            )
        if backend not in items:
            items.append(backend)
    if not items:
        raise ValueError("At least one benchmark backend is required")
    if "cpu" in items:
        items.remove("cpu")
        items.insert(0, "cpu")
    return items


def replay_command(fixture, summary_path, mode="fast"):
    return [
        "ros2",
        "launch",
        "easy_perception_deployment",
        "replay.launch.py",
        f"fixture:={Path(fixture).expanduser()}",
        f"mode:={mode}",
        f"summary_output:={Path(summary_path).expanduser()}",
    ]


def _read_summary(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "summary is not a JSON object"
    return payload, ""


def semantic_signature(summary):
    lifecycle = summary.get("object_lifecycle") or {}
    geometry = summary.get("geometry_quality") or {}
    return {
        "result": summary.get("result"),
        "completed_result_count": summary.get("completed_result_count"),
        "stable_ids": lifecycle.get("stable_ids", []),
        "lost_track_ids": lifecycle.get("lost_track_ids", []),
        "geometry_quality": geometry,
        "stale_result_count": summary.get("stale_result_count"),
    }


def performance_fields(summary):
    perf = summary.get("performance") or {}
    return {
        "resolved_backend": perf.get("execution_backend"),
        "inference_latency_min_ms": perf.get("inference_latency_min_ms"),
        "inference_latency_avg_ms": perf.get("inference_latency_avg_ms"),
        "inference_latency_max_ms": perf.get("inference_latency_max_ms"),
        "inference_rate_hz": perf.get("inference_rate_hz"),
        "observation_rate_hz": perf.get("observation_rate_hz"),
    }


def run_backend(backend, fixture, gpu_index=0, timeout=180.0):
    with tempfile.TemporaryDirectory(prefix=f"epd-benchmark-{backend}-") as temp_dir:
        summary_path = Path(temp_dir) / "summary.json"
        env = os.environ.copy()
        env["EPD_EXECUTION_BACKEND"] = backend
        env["EPD_GPU_INDEX"] = str(max(0, int(gpu_index)))
        started = time.monotonic()
        try:
            completed = subprocess.run(
                replay_command(fixture, summary_path),
                capture_output=True,
                text=True,
                check=False,
                timeout=float(timeout),
                env=env,
            )
            wall_seconds = time.monotonic() - started
        except FileNotFoundError as exc:
            return {
                "backend": backend,
                "status": "UNAVAILABLE",
                "error": str(exc),
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "backend": backend,
                "status": "FAILED",
                "wall_seconds": time.monotonic() - started,
                "error": f"benchmark timed out after {exc.timeout}s",
            }

        summary, summary_error = _read_summary(summary_path)
        record = {
            "backend": backend,
            "status": "PASS" if completed.returncode == 0 else "FAILED",
            "returncode": completed.returncode,
            "wall_seconds": round(wall_seconds, 6),
            "stderr_tail": completed.stderr[-2000:].strip(),
        }
        if summary is None:
            record["status"] = "FAILED"
            record["error"] = "replay summary unavailable: " + summary_error
            record["stdout_tail"] = completed.stdout[-2000:].strip()
            return record

        record["replay_result"] = summary.get("result")
        record["failures"] = summary.get("failures", [])
        record["performance"] = performance_fields(summary)
        record["semantic_signature"] = semantic_signature(summary)
        if summary.get("result") != "PASS":
            record["status"] = "FAILED"
        return record


def compare_with_cpu(records):
    cpu = next(
        (record for record in records if record.get("backend") == "cpu"),
        None,
    )
    if cpu is None or cpu.get("status") != "PASS":
        for record in records:
            record["equivalent_to_cpu"] = None
        return

    baseline = cpu.get("semantic_signature")
    cpu["equivalent_to_cpu"] = True
    for record in records:
        if record is cpu:
            continue
        if record.get("status") != "PASS":
            record["equivalent_to_cpu"] = False
            continue
        record["equivalent_to_cpu"] = (
            record.get("semantic_signature") == baseline
        )
        if not record["equivalent_to_cpu"]:
            record.setdefault("warnings", []).append(
                "Accelerated replay passed its own acceptance checks but did not "
                "match the CPU semantic summary exactly. Review before adoption."
            )


def build_report(backends, fixture, gpu_index=0, timeout=180.0):
    records = [
        run_backend(backend, fixture, gpu_index=gpu_index, timeout=timeout)
        for backend in backends
    ]
    compare_with_cpu(records)
    return {
        "schema_version": "epd_backend_benchmark/v1",
        "fixture": str(Path(fixture).expanduser()),
        "gpu_index": int(gpu_index),
        "records": records,
        "all_requested_passed": all(
            record.get("status") == "PASS" for record in records
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends",
        default="cpu,cuda",
        help="Comma-separated list: cpu,cuda,tensorrt",
    )
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    try:
        backends = parse_backend_list(args.backends)
    except ValueError as exc:
        parser.error(str(exc))

    report = build_report(
        backends,
        args.fixture,
        gpu_index=max(0, args.gpu_index),
        timeout=args.timeout,
    )
    pretty = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        destination = Path(args.output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(pretty + "\n", encoding="utf-8")
    # Keep stdout as one complete JSON line so the GUI can parse the result
    # without depending on log formatting from nested ros2 launch processes.
    print(json.dumps(report, sort_keys=True))
    return 0 if report["all_requested_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
