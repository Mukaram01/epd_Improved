"""Model truth and compatibility helpers for the EPD-3 Smart Model Manager."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

MODE_NAMES = {
    0: "Classification",
    1: "Counting",
    2: "Color-Matching",
    3: "Localization",
    4: "Tracking",
}

SUPPORTED_INPUT_ELEMENT_TYPES = {1, 2}  # ONNX float32 and uint8.
SUPPORTED_OUTPUT_ELEMENT_TYPES = {1}  # EPD currently reads output tensors as float32.


class ModelInspectionError(RuntimeError):
    """Raised when the runtime ONNX inspector cannot provide trustworthy data."""


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA256 digest for a local file."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_label_list(path):
    """Read non-empty UTF-8 labels while preserving order."""
    if not path:
        return [], "Label list is not configured."
    candidate = Path(path)
    if not candidate.is_file():
        return [], f"Label list file does not exist: {candidate}"
    try:
        labels = [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError) as exc:
        return [], f"Unable to read label list as UTF-8: {exc}"
    labels = [label for label in labels if label]
    if not labels:
        return [], "Label list contains no non-empty class names."
    return labels, ""


def load_model_library(package_root):
    """Load the bundled trusted-model catalog."""
    path = Path(package_root) / "data" / "model" / "model_library.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    models = data.get("models", []) if isinstance(data, dict) else []
    return [entry for entry in models if isinstance(entry, dict)]


def find_library_entry(library, digest):
    """Identify a trusted model by exact SHA256, never by filename alone."""
    digest = str(digest or "").lower()
    for entry in library:
        if str(entry.get("sha256", "")).lower() == digest:
            return entry
    return None


def precision_profile(output_count):
    """Map EPD's output-count contract to its current precision levels."""
    if output_count == 1:
        return {
            "precision_level": 1,
            "task": "Image classification (legacy P1)",
            "supported_modes": [],
            "deploy_supported": False,
        }
    if output_count == 3:
        return {
            "precision_level": 2,
            "task": "Object detection",
            "supported_modes": [0, 1, 2],
            "deploy_supported": True,
        }
    if output_count == 4:
        return {
            "precision_level": 3,
            "task": "Instance segmentation",
            "supported_modes": [0, 1, 2, 3, 4],
            "deploy_supported": True,
        }
    return {
        "precision_level": None,
        "task": "Unknown / incompatible",
        "supported_modes": [],
        "deploy_supported": False,
    }


def _direct_inspector_candidates():
    override = os.getenv("EPD_MODEL_INSPECTOR_PATH", "").strip()
    if override:
        yield [override]

    for prefix in os.getenv("AMENT_PREFIX_PATH", "").split(os.pathsep):
        prefix = prefix.strip()
        if not prefix:
            continue
        candidate = (
            Path(prefix)
            / "lib"
            / "easy_perception_deployment"
            / "epd_model_inspector"
        )
        if candidate.is_file() and os.access(candidate, os.X_OK):
            yield [str(candidate)]


def _inspector_commands(model_path):
    custom = os.getenv("EPD_MODEL_INSPECTOR_CMD", "").strip()
    if custom:
        yield shlex.split(custom) + [str(model_path)]

    for candidate in _direct_inspector_candidates():
        yield candidate + [str(model_path)]

    if shutil.which("ros2"):
        yield [
            "ros2",
            "run",
            "easy_perception_deployment",
            "epd_model_inspector",
            str(model_path),
        ]


def _parse_inspector_output(stdout):
    for line in reversed(str(stdout or "").splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "valid" in payload:
            return payload
    return None


def run_runtime_inspector(model_path, timeout_sec=10):
    """Inspect with the same vendored ONNX Runtime used by EPD deployment."""
    attempts = []
    for command in _inspector_commands(model_path):
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_sec,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            attempts.append(f"{' '.join(command[:3])}: {exc}")
            continue

        payload = _parse_inspector_output(result.stdout)
        if payload is not None:
            payload["command"] = command
            payload["returncode"] = result.returncode
            return payload

        detail = (result.stderr or result.stdout or "no JSON output").strip()
        attempts.append(f"{' '.join(command[:3])}: {detail[-240:]}")

    detail = "; ".join(attempts[-3:]) if attempts else "inspector executable not found"
    raise ModelInspectionError(detail)


def _metadata_from_trusted_entry(entry, model_path):
    """Use catalog metadata only after the file hash exactly matches."""
    rank = int(entry.get("input_rank", 0) or 0)
    layout = str(entry.get("input_layout", "") or "")
    output_count = int(entry.get("output_count", 0) or 0)
    return {
        "valid": True,
        "model_path": str(model_path),
        "input_count": 1,
        "output_count": output_count,
        "inputs": [
            {
                "name": "catalog-known input",
                "tensor": True,
                "element_type": 1,
                "rank": rank,
                "shape": [],
                "catalog_layout": layout,
            }
        ],
        "outputs": [
            {
                "name": f"catalog-known output {index + 1}",
                "tensor": True,
                "element_type": 1,
                "rank": None,
                "shape": [],
            }
            for index in range(output_count)
        ],
        "trusted_catalog": True,
    }


def _validate_input(metadata, blockers, warnings):
    input_count = int(metadata.get("input_count", 0) or 0)
    inputs = metadata.get("inputs", []) or []
    if input_count != 1:
        blockers.append(
            f"EPD currently requires exactly one model input; this model has {input_count}."
        )
        return
    if not inputs:
        blockers.append("Inspector returned no input tensor metadata.")
        return

    first = inputs[0]
    if not first.get("tensor", False):
        blockers.append("The first model input is not a tensor.")
        return

    element_type = first.get("element_type")
    if element_type not in SUPPORTED_INPUT_ELEMENT_TYPES:
        blockers.append(
            "EPD input must be float32 or uint8; "
            f"the model declares ONNX element type {element_type}."
        )

    rank = first.get("rank")
    if rank not in (3, 4):
        blockers.append(
            f"EPD requires a rank-3 CHW or rank-4 NCHW/NHWC image input; rank is {rank}."
        )
        return

    if metadata.get("trusted_catalog"):
        return

    shape = first.get("shape", []) or []
    if len(shape) != rank:
        warnings.append("Input shape metadata is incomplete; channel layout was not proven.")
        return

    if rank == 3:
        channel = shape[0]
        if channel not in (3, -1):
            blockers.append(
                "Rank-3 input is not compatible with EPD CHW RGB feeding: "
                f"expected channel dimension 3/dynamic, got {channel}."
            )
        return

    dim1 = shape[1]
    dim3 = shape[3]
    if dim3 == 3 and dim1 != 3:
        return
    if dim1 in (3, -1):
        return
    blockers.append(
        "Rank-4 input is not compatible with EPD RGB feeding: expected channel "
        f"dimension at index 1 or 3, got shape {shape}."
    )


def _validate_outputs(metadata, blockers):
    output_count = int(metadata.get("output_count", 0) or 0)
    profile = precision_profile(output_count)
    if not profile["deploy_supported"]:
        if output_count == 1:
            blockers.append(
                "Precision Level 1 deployment is deprecated in the current EPD runtime. "
                "Use a P2 (3-output) or P3 (4-output) model."
            )
        else:
            blockers.append(
                "EPD identifies models by output count: P2 requires 3 outputs and "
                f"P3 requires 4; this model has {output_count}."
            )

    for index, output in enumerate(metadata.get("outputs", []) or []):
        if not output.get("tensor", False):
            blockers.append(f"Model output {index + 1} is not a tensor.")
            continue
        element_type = output.get("element_type")
        if element_type not in SUPPORTED_OUTPUT_ELEMENT_TYPES:
            blockers.append(
                f"Model output {index + 1} is ONNX element type {element_type}; "
                "EPD currently reads outputs as float32."
            )
    return profile


def _validate_labels(entry, labels, label_error, package_root, blockers, warnings):
    if label_error:
        blockers.append(label_error)
        return {
            "count": 0,
            "state": "blocked",
            "detail": label_error,
        }

    state = "valid"
    detail = f"{len(labels)} labels loaded."
    canonical_name = entry.get("canonical_labels") if entry else None
    expected_count = entry.get("label_count") if entry else None

    if expected_count is not None and len(labels) != int(expected_count):
        blockers.append(
            f"Trusted model expects {expected_count} labels, but the selected list has "
            f"{len(labels)}."
        )
        state = "blocked"
        detail = f"Count mismatch: expected {expected_count}, got {len(labels)}."

    if canonical_name:
        canonical_path = Path(package_root) / "data" / "label_list" / canonical_name
        canonical, canonical_error = read_label_list(canonical_path)
        if not canonical_error:
            if labels != canonical:
                mismatch = next(
                    (
                        index
                        for index, pair in enumerate(zip(labels, canonical))
                        if pair[0] != pair[1]
                    ),
                    min(len(labels), len(canonical)),
                )
                blockers.append(
                    "Label order/content does not match the trusted model's canonical "
                    f"{canonical_name} list (first mismatch near index {mismatch})."
                )
                state = "blocked"
                detail = f"Does not match canonical {canonical_name} order."
            elif state != "blocked":
                state = "verified"
                detail = f"Exact canonical {canonical_name} order verified."
    elif entry is None:
        warnings.append(
            "Custom ONNX models do not encode a reliable class-label order for EPD. "
            "The label list is structurally valid, but its order cannot be proven."
        )
        detail = f"{len(labels)} labels; order not provable from this ONNX file."

    return {
        "count": len(labels),
        "state": state,
        "detail": detail,
    }


def inspect_deployment_model(model_path, label_path, usecase_mode, package_root):
    """Return a conservative preflight verdict for the current Deploy selection."""
    package_root = Path(package_root)
    model_path = Path(model_path) if model_path else Path()
    label_path = Path(label_path) if label_path else Path()
    blockers = []
    warnings = []
    mode = int(usecase_mode)
    mode_name = MODE_NAMES.get(mode, f"Mode {mode}")

    result = {
        "status": "blocked",
        "summary": "Model is not ready.",
        "model_path": str(model_path),
        "label_path": str(label_path),
        "mode": mode,
        "mode_name": mode_name,
        "blockers": blockers,
        "warnings": warnings,
        "library_entry": None,
        "inspection_source": "none",
        "metadata": None,
        "labels": None,
        "precision_level": None,
        "task": "Unknown",
        "supported_modes": [],
        "recommended_modes": [],
    }

    if not str(model_path) or not model_path.is_file():
        blockers.append(f"ONNX model file does not exist: {model_path or '(not set)'}")
        return result
    if model_path.suffix.lower() != ".onnx":
        blockers.append("EPD deployment requires a .onnx model file.")
        return result
    if model_path.stat().st_size <= 0:
        blockers.append("ONNX model file is empty.")
        return result

    try:
        digest = sha256_file(model_path)
    except OSError as exc:
        blockers.append(f"Unable to hash model file: {exc}")
        return result
    result["sha256"] = digest

    library = load_model_library(package_root)
    entry = find_library_entry(library, digest)
    result["library_entry"] = entry

    metadata = None
    try:
        metadata = run_runtime_inspector(model_path)
        result["inspection_source"] = "EPD ONNX Runtime inspector"
    except ModelInspectionError as exc:
        if entry is not None:
            metadata = _metadata_from_trusted_entry(entry, model_path)
            result["inspection_source"] = "trusted SHA256 catalog"
            warnings.append(
                "Runtime inspector is unavailable, so this exact trusted model is "
                f"validated from its catalog checksum. Inspector detail: {exc}"
            )
        else:
            blockers.append(
                "Custom model could not be inspected with EPD's ONNX Runtime. Rebuild "
                "and source the workspace so epd_model_inspector is available. "
                f"Detail: {exc}"
            )
            return result

    result["metadata"] = metadata
    if not metadata.get("valid", False):
        blockers.append(
            "ONNX Runtime could not load the model: "
            f"{metadata.get('error', 'unknown model error')}"
        )
        return result

    _validate_input(metadata, blockers, warnings)
    profile = _validate_outputs(metadata, blockers)
    result["precision_level"] = profile["precision_level"]
    result["task"] = profile["task"]
    result["supported_modes"] = [MODE_NAMES[item] for item in profile["supported_modes"]]

    if entry is not None:
        result["task"] = str(entry.get("task") or result["task"])
        result["recommended_modes"] = list(entry.get("recommended_modes", []) or [])
    else:
        result["recommended_modes"] = list(result["supported_modes"])

    if mode not in profile["supported_modes"]:
        if profile["precision_level"] == 2 and mode in (3, 4):
            blockers.append(
                f"{mode_name} requires a Precision Level 3 / 4-output model. "
                "Select Mask R-CNN or another compatible P3 model."
            )
        elif profile["deploy_supported"]:
            blockers.append(
                f"{mode_name} is not supported by this model's EPD precision level."
            )

    labels, label_error = read_label_list(label_path)
    result["labels"] = _validate_labels(
        entry,
        labels,
        label_error,
        package_root,
        blockers,
        warnings,
    )

    if blockers:
        result["summary"] = blockers[0]
        result["status"] = "blocked"
    else:
        result["status"] = "ready"
        if entry is not None:
            result["summary"] = (
                f"Ready: trusted {entry.get('name', model_path.name)} is compatible "
                f"with {mode_name}."
            )
        else:
            result["summary"] = (
                f"Ready: ONNX Runtime verified a compatible P{profile['precision_level']} "
                f"model for {mode_name}."
            )
    return result


def library_install_state(package_root, entry):
    """Return installed/missing/corrupt state for one trusted catalog entry."""
    model_path = Path(package_root) / "data" / "model" / str(entry.get("filename", ""))
    if not model_path.is_file():
        return "missing", str(model_path)
    try:
        digest = sha256_file(model_path)
    except OSError:
        return "unreadable", str(model_path)
    if digest.lower() != str(entry.get("sha256", "")).lower():
        return "checksum mismatch", str(model_path)
    return "installed", str(model_path)
