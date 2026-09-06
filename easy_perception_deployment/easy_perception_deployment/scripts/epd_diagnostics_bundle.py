#!/usr/bin/env python3
"""Collect a portable, read-only EPD diagnostics bundle.

The collector is intentionally best-effort: missing ROS/Docker/NVIDIA tooling is
reported in the manifest rather than making bundle creation fail. By default
absolute home/workspace paths are redacted from captured text.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "epd_diagnostics_bundle/v1"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def command_result(command, timeout=8.0, env=None):
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
            "command": command,
            "returncode": completed.returncode,
            "duration_s": round(time.monotonic() - started, 3),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "returncode": None,
            "duration_s": round(time.monotonic() - started, 3),
            "error": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "duration_s": round(time.monotonic() - started, 3),
            "error": f"timeout after {exc.timeout}s",
        }
    except Exception as exc:  # diagnostic collection must remain best-effort
        return {
            "command": command,
            "returncode": None,
            "duration_s": round(time.monotonic() - started, 3),
            "error": str(exc),
        }


def redactor(include_paths=False):
    if include_paths:
        return lambda value: value
    replacements = []
    for candidate, replacement in (
        (str(Path.home()), "<HOME>"),
        (os.environ.get("USER", ""), "<USER>"),
    ):
        candidate = str(candidate or "").strip()
        if candidate:
            replacements.append((candidate, replacement))

    def redact(value):
        text = ANSI_RE.sub("", str(value or ""))
        for source, replacement in replacements:
            text = text.replace(source, replacement)
        return text

    return redact


def sanitize_result(result, redact):
    cleaned = dict(result)
    cleaned["command"] = [redact(item) for item in result.get("command", [])]
    for key in ("stdout", "stderr", "error"):
        if key in cleaned:
            cleaned[key] = redact(cleaned[key])
    return cleaned


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_text_file(source, destination, redact):
    try:
        content = source.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"source": str(source), "copied": False, "error": str(exc)}
    write_text(destination, redact(content))
    return {"source": str(source), "copied": True, "size": len(content)}


def package_root_from_script():
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


def collect(package_root, bundle_dir, profile_path=None, include_paths=False):
    redact = redactor(include_paths)
    manifest = {
        "schema_version": SCHEMA,
        "created_utc": utc_now(),
        "read_only_collection": True,
        "paths_redacted": not include_paths,
        "package_root": redact(str(package_root)),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "ros_distro": os.environ.get("ROS_DISTRO", ""),
        },
        "environment": {
            key: redact(os.environ.get(key, ""))
            for key in (
                "ROS_DISTRO",
                "ROS_DOMAIN_ID",
                "RMW_IMPLEMENTATION",
                "EPD_EXECUTION_BACKEND",
                "EPD_GPU_INDEX",
                "EPD_PACKAGE_ROOT",
            )
        },
        "files": [],
        "commands": {},
    }

    config_dir = package_root / "config"
    for name in (
        "session_config.json",
        "usecase_config.json",
        "input_image_topic.json",
    ):
        source = config_dir / name
        if source.is_file():
            record = copy_text_file(
                source,
                bundle_dir / "config" / name,
                redact,
            )
            record["source"] = redact(record["source"])
            manifest["files"].append(record)
        else:
            manifest["files"].append({
                "source": redact(str(source)),
                "copied": False,
                "error": "config file not found",
            })

    if profile_path:
        profile = Path(profile_path).expanduser().resolve()
        if profile.is_file():
            record = copy_text_file(
                profile,
                bundle_dir / "profile" / profile.name,
                redact,
            )
            record["source"] = redact(record["source"])
            manifest["files"].append(record)
        else:
            manifest["files"].append({
                "source": redact(str(profile)),
                "copied": False,
                "error": "profile file not found",
            })

    logs_dir = package_root / "gui" / "scripts" / "logs"
    if logs_dir.is_dir():
        for source in sorted(logs_dir.glob("*.log"))[-8:]:
            record = copy_text_file(
                source,
                bundle_dir / "logs" / source.name,
                redact,
            )
            record["source"] = redact(record["source"])
            manifest["files"].append(record)

    commands = {
        "uname": ["uname", "-a"],
        "docker_version": ["docker", "--version"],
        "nvidia_smi": ["nvidia-smi", "-L"],
        "ros2_nodes": ["ros2", "node", "list"],
        "ros2_topics": ["ros2", "topic", "list", "-t"],
        "backend_probe": [
            "ros2", "run", "easy_perception_deployment", "epd_backend_probe"
        ],
        "inference_diagnostics": [
            "ros2", "topic", "echo",
            "/easy_perception_deployment/inference_diagnostics",
            "--once",
        ],
        "tracking_sample": [
            "ros2", "topic", "echo",
            "/easy_perception_deployment/epd_tracking_output",
            "--once",
        ],
        "workcell_status_sample": [
            "ros2", "topic", "echo",
            "/workcell_studio/epd_connector_status",
            "--once",
        ],
    }
    short_timeout = {
        "inference_diagnostics": 3.0,
        "tracking_sample": 3.0,
        "workcell_status_sample": 3.0,
    }
    for name, command in commands.items():
        result = command_result(command, timeout=short_timeout.get(name, 8.0))
        clean = sanitize_result(result, redact)
        manifest["commands"][name] = {
            key: value for key, value in clean.items()
            if key not in ("stdout", "stderr")
        }
        body = []
        body.append("$ " + " ".join(clean.get("command", [])))
        body.append(f"returncode: {clean.get('returncode')}")
        body.append(f"duration_s: {clean.get('duration_s')}")
        if clean.get("error"):
            body.append("error: " + clean["error"])
        if clean.get("stdout"):
            body.append("\n--- stdout ---\n" + clean["stdout"].rstrip())
        if clean.get("stderr"):
            body.append("\n--- stderr ---\n" + clean["stderr"].rstrip())
        write_text(bundle_dir / "commands" / f"{name}.txt", "\n".join(body) + "\n")

    readme = (
        "EPD diagnostics bundle\n"
        "======================\n\n"
        "Generated read-only by epd_diagnostics_bundle.py.\n"
        "Missing commands/topics are evidence, not bundle-generation failures.\n"
        f"Paths redacted: {not include_paths}\n"
        "This bundle is diagnostic evidence, not a safety certificate.\n"
    )
    write_text(bundle_dir / "README.txt", readme)
    write_text(
        bundle_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def zip_directory(source_dir, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", default=str(package_root_from_script()))
    parser.add_argument("--profile")
    parser.add_argument("--output")
    parser.add_argument(
        "--include-paths",
        action="store_true",
        help="Do not redact HOME/user strings from captured text",
    )
    args = parser.parse_args()

    package_root = Path(args.package_root).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output or f"epd_diagnostics_{stamp}.zip").expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="epd-diagnostics-") as temp:
        bundle_dir = Path(temp) / "epd_diagnostics"
        manifest = collect(
            package_root,
            bundle_dir,
            profile_path=args.profile,
            include_paths=args.include_paths,
        )
        zip_directory(bundle_dir, output)

    print(json.dumps({
        "schema_version": SCHEMA,
        "output": str(output),
        "created_utc": manifest["created_utc"],
        "paths_redacted": manifest["paths_redacted"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
