#!/usr/bin/env python3
"""Run a repeatable, read-only EPD release/demo acceptance checklist."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "epd_release_acceptance/v1"


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def check(name, status, evidence, remediation=""):
    return {
        "name": name,
        "status": status,
        "evidence": evidence,
        "remediation": remediation,
    }


def run(command, timeout=15.0, env=None):
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "duration_s": round(time.monotonic() - started, 3),
        }
    except FileNotFoundError as exc:
        return {"returncode": None, "error": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {"returncode": None, "error": f"timeout after {exc.timeout}s"}


def default_package_root():
    configured = os.environ.get("EPD_PACKAGE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    source_candidate = Path(__file__).resolve().parents[1]
    if (source_candidate / "config" / "session_config.json").is_file():
        return source_candidate

    cwd = Path.cwd().resolve()
    if (cwd / "config" / "session_config.json").is_file():
        return cwd

    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(
            get_package_share_directory("easy_perception_deployment")
        ).resolve()
    except Exception:
        return source_candidate


def load_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, "root value is not a JSON object"
    return value, ""


def resolve_asset(package_root, configured):
    path = Path(str(configured or "")).expanduser()
    if not path.is_absolute():
        path = package_root / path
    return path.resolve()


def static_checks(package_root):
    results = []
    config_dir = package_root / "config"
    configs = {}
    for filename in (
        "session_config.json",
        "usecase_config.json",
        "input_image_topic.json",
    ):
        path = config_dir / filename
        data, error = load_json(path)
        if data is None:
            results.append(check(
                filename,
                "FAIL",
                error,
                f"Restore or repair {path}",
            ))
        else:
            configs[filename] = data
            results.append(check(filename, "PASS", "valid JSON object"))

    session = configs.get("session_config.json") or {}
    model = resolve_asset(package_root, session.get("path_to_model"))
    labels = resolve_asset(package_root, session.get("path_to_label_list"))
    results.append(check(
        "ONNX model asset",
        "PASS" if model.is_file() else "FAIL",
        str(model),
        "Select/download a valid ONNX model in Deploy" if not model.is_file() else "",
    ))
    results.append(check(
        "Label list asset",
        "PASS" if labels.is_file() else "FAIL",
        str(labels),
        "Select a label list matching the model" if not labels.is_file() else "",
    ))

    backend = str(session.get("execution_backend", "auto")).lower()
    results.append(check(
        "Execution backend config",
        "PASS" if backend in ("auto", "cpu", "cuda", "tensorrt") else "FAIL",
        backend,
        "Use auto, cpu, cuda, or tensorrt" if backend not in (
            "auto", "cpu", "cuda", "tensorrt") else "",
    ))

    ros_distro = os.environ.get("ROS_DISTRO", "")
    results.append(check(
        "ROS distribution",
        "PASS" if ros_distro == "humble" else "WARN",
        ros_distro or "ROS_DISTRO is not set",
        "Source ROS 2 Humble for the supported baseline" if ros_distro != "humble" else "",
    ))

    expected = (
        package_root / "launch" / "run.launch.py",
        package_root / "launch" / "replay.launch.py",
        package_root / "launch" / "workcell_contract.launch.py",
    )
    missing = [str(path) for path in expected if not path.is_file()]
    results.append(check(
        "Release launch files",
        "PASS" if not missing else "FAIL",
        "all present" if not missing else "; ".join(missing),
    ))
    return results


def ros_checks():
    results = []
    ros2 = run(["ros2", "node", "list"], timeout=8.0)
    if ros2.get("returncode") is None:
        return [check(
            "ROS 2 CLI",
            "WARN",
            ros2.get("error", "ros2 unavailable"),
            "Source ROS 2 Humble before live acceptance",
        )]
    results.append(check(
        "ROS 2 CLI",
        "PASS" if ros2["returncode"] == 0 else "WARN",
        ros2.get("stdout") or ros2.get("stderr") or "ROS graph queried",
    ))

    topics = run(["ros2", "topic", "list", "-t"], timeout=8.0)
    topic_text = topics.get("stdout", "")
    for name, topic in (
        ("RGB topic", "/camera/camera/color/image_raw"),
        ("Aligned depth topic", "/camera/camera/aligned_depth_to_color/image_raw"),
        ("CameraInfo topic", "/camera/camera/color/camera_info"),
        ("Inference diagnostics", "/easy_perception_deployment/inference_diagnostics"),
    ):
        present = topic in topic_text
        results.append(check(
            name,
            "PASS" if present else "WARN",
            topic + (" detected" if present else " not currently detected"),
        ))
    return results


def backend_probe_check():
    result = run(
        ["ros2", "run", "easy_perception_deployment", "epd_backend_probe"],
        timeout=10.0,
    )
    if result.get("returncode") is None:
        return check(
            "Backend capability probe",
            "WARN",
            result.get("error", "probe unavailable"),
            "Build/source EPD before backend acceptance",
        )
    status = "PASS" if result["returncode"] == 0 else "WARN"
    evidence = result.get("stdout") or result.get("stderr") or "probe returned no text"
    return check("Backend capability probe", status, evidence)


def replay_check(package_root, timeout):
    fixture = package_root / "fixtures" / "p8_tracking.json"
    if not fixture.is_file():
        return check("Deterministic replay", "FAIL", f"fixture missing: {fixture}")
    with tempfile.TemporaryDirectory(prefix="epd-release-replay-") as temp:
        summary = Path(temp) / "summary.json"
        result = run(
            [
                "ros2", "launch", "easy_perception_deployment", "replay.launch.py",
                f"fixture:={fixture}",
                "mode:=fast",
                f"summary_output:={summary}",
            ],
            timeout=timeout,
        )
        if result.get("returncode") is None:
            return check(
                "Deterministic replay",
                "FAIL",
                result.get("error", "replay unavailable"),
            )
        payload, error = load_json(summary) if summary.is_file() else (None, "summary missing")
        if result["returncode"] == 0 and payload and payload.get("result") == "PASS":
            perf = payload.get("performance") or {}
            return check(
                "Deterministic replay",
                "PASS",
                json.dumps({
                    "result": payload.get("result"),
                    "completed_result_count": payload.get("completed_result_count"),
                    "stable_ids": (payload.get("object_lifecycle") or {}).get("stable_ids", []),
                    "execution_backend": perf.get("execution_backend"),
                    "inference_latency_avg_ms": perf.get("inference_latency_avg_ms"),
                }, sort_keys=True),
            )
        return check(
            "Deterministic replay",
            "FAIL",
            error or json.dumps(payload or {}) or result.get("stderr", "replay failed"),
        )


def overall_status(results):
    statuses = {item["status"] for item in results}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root",
        default=str(default_package_root()),
        help="EPD share/source directory containing config, data, launch and fixtures",
    )
    parser.add_argument("--with-ros", action="store_true")
    parser.add_argument("--with-replay", action="store_true")
    parser.add_argument("--replay-timeout", type=float, default=180.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    package_root = Path(args.package_root).expanduser().resolve()
    results = static_checks(package_root)
    if args.with_ros or args.with_replay:
        results.extend(ros_checks())
        results.append(backend_probe_check())
    if args.with_replay:
        results.append(replay_check(package_root, args.replay_timeout))

    report = {
        "schema_version": SCHEMA,
        "created_utc": utc_now(),
        "package_root": str(package_root),
        "with_ros": bool(args.with_ros or args.with_replay),
        "with_replay": bool(args.with_replay),
        "status": overall_status(results),
        "checks": results,
        "note": "Acceptance evidence is not a safety certificate.",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
