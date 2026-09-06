"""Versioned EPD perception profiles for reproducible deploy sessions."""

from __future__ import annotations

import json
import os
import re
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from windows.model_manager import sha256_file

_PROFILE_SCHEMA_VERSION = 1
_PROFILE_SUFFIX = ".epd-profile.json"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class ProfileError(ValueError):
    """Raised when a profile cannot be validated or applied safely."""


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProfileError(f"Unable to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ProfileError(f"{path} must contain a JSON object")
    return data


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _clean_name(name):
    name = " ".join(str(name or "").strip().split())
    if not name:
        raise ProfileError("Profile name must not be empty")
    if len(name) > 80:
        raise ProfileError("Profile name must be 80 characters or fewer")
    return name


def _slug(name):
    clean = _SAFE_NAME_RE.sub("-", _clean_name(name)).strip("-._")
    return (clean or "profile")[:80]


def _resolve_asset(package_root, configured_path):
    path = Path(str(configured_path or "")).expanduser()
    if not path.is_absolute():
        path = Path(package_root) / path
    return path.resolve()


def _asset_metadata(package_root, configured_path):
    if not configured_path:
        return {"path": "", "sha256": "", "size": 0}
    resolved = _resolve_asset(package_root, configured_path)
    if not resolved.is_file():
        return {
            "path": str(configured_path),
            "sha256": "",
            "size": 0,
            "missing_at_capture": True,
        }
    return {
        "path": str(configured_path),
        "sha256": sha256_file(resolved),
        "size": resolved.stat().st_size,
        "basename": resolved.name,
    }


def capture_profile(package_root, name, description=""):
    """Capture the current three EPD deploy config files into one profile."""
    package_root = Path(package_root)
    session = _read_json(package_root / "config" / "session_config.json")
    usecase = _read_json(package_root / "config" / "usecase_config.json")
    input_topic = _read_json(package_root / "config" / "input_image_topic.json")
    profile_name = _clean_name(name)
    timestamp = _utc_now()
    return {
        "profile_schema_version": _PROFILE_SCHEMA_VERSION,
        "name": profile_name,
        "description": str(description or "").strip(),
        "created_utc": timestamp,
        "updated_utc": timestamp,
        "epd": {
            "session_config": deepcopy(session),
            "usecase_config": deepcopy(usecase),
            "input_image_topic_config": deepcopy(input_topic),
        },
        "assets": {
            "model": _asset_metadata(package_root, session.get("path_to_model")),
            "labels": _asset_metadata(
                package_root,
                session.get("path_to_label_list"),
            ),
        },
    }


def normalize_profile(data):
    """Validate the profile envelope without relying on the current workstation."""
    if not isinstance(data, dict):
        raise ProfileError("Profile must contain a JSON object")
    if data.get("profile_schema_version") != _PROFILE_SCHEMA_VERSION:
        raise ProfileError(
            "Unsupported profile schema version: "
            f"{data.get('profile_schema_version')!r}; expected {_PROFILE_SCHEMA_VERSION}"
        )
    normalized = deepcopy(data)
    normalized["name"] = _clean_name(normalized.get("name"))
    normalized["description"] = str(normalized.get("description", "")).strip()
    epd = normalized.get("epd")
    if not isinstance(epd, dict):
        raise ProfileError("Profile is missing the epd configuration object")
    required = (
        "session_config",
        "usecase_config",
        "input_image_topic_config",
    )
    for key in required:
        if not isinstance(epd.get(key), dict):
            raise ProfileError(f"Profile epd.{key} must contain a JSON object")

    session = epd["session_config"]
    for key in ("path_to_model", "path_to_label_list"):
        if not str(session.get(key, "")).strip():
            raise ProfileError(f"Profile session_config.{key} must not be empty")
    if str(session.get("visualizeFlag", "")) not in ("visualize", "robot"):
        raise ProfileError("Profile visualizeFlag must be 'visualize' or 'robot'")
    if str(session.get("useCPU", "")) not in ("CPU", "GPU"):
        raise ProfileError("Profile useCPU must be 'CPU' or 'GPU'")
    confidence = float(session.get("confidence_threshold", 0.5))
    if confidence < 0.0 or confidence > 1.0:
        raise ProfileError("Profile confidence_threshold must be between 0 and 1")
    if int(session.get("max_detections", 100)) < 0:
        raise ProfileError("Profile max_detections must be >= 0")

    mode = int(epd["usecase_config"].get("usecase_mode", -1))
    if mode not in range(5):
        raise ProfileError("Profile usecase_mode must be an integer from 0 to 4")
    if mode == 1:
        classes = epd["usecase_config"].get("class_list")
        if not isinstance(classes, list) or not classes:
            raise ProfileError("Counting profile requires a non-empty class_list")
    if mode == 2 and not str(
        epd["usecase_config"].get("path_to_color_template", "")
    ).strip():
        raise ProfileError("Color-Matching profile requires path_to_color_template")
    if mode == 4 and not str(epd["usecase_config"].get("track_type", "")).strip():
        raise ProfileError("Tracking profile requires track_type")

    topic = str(
        epd["input_image_topic_config"].get("input_image_topic", "")
    ).strip()
    if not topic:
        raise ProfileError("Profile input_image_topic must not be empty")
    return normalized


def _candidate_asset(package_root, configured_path, kind):
    configured = Path(str(configured_path)).expanduser()
    if configured.is_absolute() and configured.is_file():
        return configured.resolve()
    direct = _resolve_asset(package_root, configured_path)
    if direct.is_file():
        return direct
    subdir = "model" if kind == "model" else "label_list"
    candidate = Path(package_root) / "data" / subdir / configured.name
    return candidate.resolve() if candidate.is_file() else None


def resolve_profile_assets(profile, package_root):
    """Resolve portable model/label paths and verify captured hashes when present."""
    profile = normalize_profile(profile)
    package_root = Path(package_root).resolve()
    session = profile["epd"]["session_config"]
    warnings = []
    for key, kind in (("path_to_model", "model"), ("path_to_label_list", "labels")):
        configured = session[key]
        resolved = _candidate_asset(package_root, configured, "model" if kind == "model" else "labels")
        if resolved is None:
            raise ProfileError(
                f"Profile {kind} asset is missing on this workstation: {configured}"
            )
        expected = str(profile.get("assets", {}).get(kind, {}).get("sha256", "")).lower()
        if expected:
            actual = sha256_file(resolved).lower()
            if actual != expected:
                raise ProfileError(
                    f"Profile {kind} SHA256 mismatch for {resolved.name}; refusing to "
                    "silently substitute a different asset"
                )
        original = _resolve_asset(package_root, configured)
        if resolved != original:
            relative = resolved
            try:
                relative = resolved.relative_to(package_root)
                session[key] = "./" + relative.as_posix()
            except ValueError:
                session[key] = str(resolved)
            warnings.append(
                f"Relocated {kind} by verified basename/hash to {session[key]}"
            )
    return profile, warnings


def apply_profile_to_files(profile, package_root):
    """Atomically replace the EPD deploy configs after profile validation."""
    profile, warnings = resolve_profile_assets(profile, package_root)
    root = Path(package_root)
    epd = profile["epd"]
    _write_json_atomic(root / "config" / "session_config.json", epd["session_config"])
    _write_json_atomic(root / "config" / "usecase_config.json", epd["usecase_config"])
    _write_json_atomic(
        root / "config" / "input_image_topic.json",
        epd["input_image_topic_config"],
    )
    return profile, warnings


def profile_summary(profile):
    profile = normalize_profile(profile)
    epd = profile["epd"]
    session = epd["session_config"]
    usecase = epd["usecase_config"]
    mode_names = ["Classification", "Counting", "Color-Matching", "Localization", "Tracking"]
    mode = int(usecase.get("usecase_mode", 0))
    return {
        "name": profile["name"],
        "description": profile.get("description", ""),
        "model": Path(str(session.get("path_to_model", ""))).name,
        "labels": Path(str(session.get("path_to_label_list", ""))).name,
        "topic": epd["input_image_topic_config"].get("input_image_topic", ""),
        "mode": mode_names[mode],
        "device": session.get("useCPU", "CPU"),
        "confidence": float(session.get("confidence_threshold", 0.5)),
        "transport": session.get("image_transport", "raw"),
        "overlay": session.get("visualizeFlag", "robot") == "visualize",
        "masks": bool(session.get("publish_detection_segmentation", True)),
    }


class ProfileStore:
    """User-level profile storage independent of the source checkout."""

    def __init__(self, directory=None):
        if directory is None:
            directory = (
                Path.home()
                / ".config"
                / "WorkcellStudio"
                / "EasyPerceptionDeployment"
                / "profiles"
            )
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.marker_path = self.directory / "known_good.json"

    def path_for_name(self, name):
        return self.directory / (_slug(name) + _PROFILE_SUFFIX)

    def save(self, profile):
        profile = normalize_profile(profile)
        path = self.path_for_name(profile["name"])
        if path.is_file():
            previous = _read_json(path)
            profile["created_utc"] = previous.get(
                "created_utc",
                profile.get("created_utc", _utc_now()),
            )
        profile["updated_utc"] = _utc_now()
        _write_json_atomic(path, profile)
        return path

    def list_profiles(self):
        records = []
        for path in sorted(self.directory.glob("*" + _PROFILE_SUFFIX)):
            try:
                profile = normalize_profile(_read_json(path))
                records.append((path, profile))
            except ProfileError:
                continue
        return records

    def load(self, path):
        return normalize_profile(_read_json(path))

    def import_profile(self, source):
        profile = self.load(source)
        return self.save(profile)

    def export_profile(self, profile_path, destination):
        profile = self.load(profile_path)
        destination = Path(destination)
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(_PROFILE_SUFFIX)
        _write_json_atomic(destination, profile)
        return destination

    def delete(self, profile_path):
        path = Path(profile_path)
        known = self.known_good_path()
        if path.exists():
            path.unlink()
        if known is not None and known.resolve() == path.resolve():
            self.clear_known_good()

    def set_known_good(self, profile_path):
        path = Path(profile_path).resolve()
        if not path.is_file():
            raise ProfileError("Known-good profile file does not exist")
        _write_json_atomic(
            self.marker_path,
            {"profile_path": str(path), "set_utc": _utc_now()},
        )

    def known_good_path(self):
        if not self.marker_path.is_file():
            return None
        try:
            marker = _read_json(self.marker_path)
        except ProfileError:
            return None
        path = Path(str(marker.get("profile_path", ""))).expanduser()
        return path if path.is_file() else None

    def clear_known_good(self):
        if self.marker_path.exists():
            self.marker_path.unlink()

    def restore_known_good(self):
        path = self.known_good_path()
        if path is None:
            raise ProfileError("No known-good profile is configured")
        return path, self.load(path)
