#!/usr/bin/env python3

"""Shared EPD config schema, validation and migration helpers."""

from __future__ import annotations

import os
from copy import deepcopy

SCHEMA_VERSION = 2

COLOR_HISTOGRAM_METRIC_CHOICES = (
    "Correlation",
    "Chi-square",
    "Intersection",
    "Bhattacharyya",
)
TRACK_TYPE_CHOICES = ("KCF", "MEDIANFLOW", "CSRT")
IMAGE_TRANSPORT_CHOICES = ("raw", "compressed")
EXECUTION_BACKEND_CHOICES = ("auto", "cpu", "cuda", "tensorrt")

USECASE_MODE_CHOICES = (0, 1, 2, 3, 4)
USECASE_MODE_LABELS = {
    0: "CLASSIFICATION",
    1: "COUNTING",
    2: "COLOR-MATCHING",
    3: "LOCALIZATION",
    4: "TRACKING",
}

DEFAULT_SESSION_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "path_to_model": "./data/model/squeezenet1.1-7.onnx",
    "path_to_label_list": "./data/label_list/imagenet_classes.txt",
    "visualizeFlag": "visualize",
    "useCPU": "CPU",
    "execution_backend": "auto",
    "execution_backend_gpu_index": 0,
    "intra_op_num_threads": 0,
    "image_transport": "raw",
    "publish_detection_segmentation": True,
    "confidence_threshold": 0.5,
    "max_detections": 100,
}

DEFAULT_USECASE_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "usecase_mode": 0,
}

DEFAULT_INPUT_TOPIC_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "input_image_topic": "/camera/camera/color/image_raw",
}


class ConfigSchemaError(ValueError):
    """Raised when EPD config schema validation fails."""


def _error(field: str, message: str) -> ConfigSchemaError:
    return ConfigSchemaError(f"{field}: {message}")


def normalize_execution_backend(value):
    normalized = str(value or "auto").strip().lower()
    aliases = {
        "gpu": "cuda",
        "nvidia": "cuda",
        "trt": "tensorrt",
        "default": "auto",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in EXECUTION_BACKEND_CHOICES:
        raise _error(
            "execution_backend",
            "invalid value (expected one of "
            + ", ".join(EXECUTION_BACKEND_CHOICES)
            + ")",
        )
    return normalized


def normalize_color_histogram_metric(metric):
    if isinstance(metric, int):
        metric_value = metric
    else:
        metric_str = str(metric).strip()
        if metric_str.isdigit():
            metric_value = int(metric_str)
        else:
            metric_lower = metric_str.lower()
            if metric_lower == "correlation":
                return "Correlation"
            if metric_lower in ("chi-square", "chisquare", "chi_square"):
                return "Chi-square"
            if metric_lower == "intersection":
                return "Intersection"
            if metric_lower == "bhattacharyya":
                return "Bhattacharyya"
            raise _error(
                "color_match_histogram_metric",
                "invalid value (expected 0-3 or one of "
                + ", ".join(COLOR_HISTOGRAM_METRIC_CHOICES)
                + ")",
            )

    metrics_by_int = {
        0: "Correlation",
        1: "Chi-square",
        2: "Intersection",
        3: "Bhattacharyya",
    }
    if metric_value in metrics_by_int:
        return metrics_by_int[metric_value]
    raise _error(
        "color_match_histogram_metric",
        "invalid value (expected 0-3)",
    )


def normalize_track_type(track_type):
    normalized = str(track_type or "").strip().upper()
    if normalized in TRACK_TYPE_CHOICES:
        return normalized
    raise _error(
        "track_type",
        "invalid value (expected one of " + ", ".join(TRACK_TYPE_CHOICES) + ")",
    )


def normalize_class_list(class_list):
    if not isinstance(class_list, list):
        raise _error("class_list", "must be a list of non-empty class names")
    normalized = []
    for class_name in class_list:
        class_name_str = str(class_name).strip()
        if class_name_str:
            normalized.append(class_name_str)
    if not normalized:
        raise _error("class_list", "must contain at least one class name")
    return normalized


def validate_existing_file_path(filepath, field_name):
    candidate_path = str(filepath or "").strip()
    if not candidate_path:
        raise _error(field_name, "path cannot be empty")
    resolved_path = os.path.abspath(os.path.expanduser(candidate_path))
    if not os.path.isfile(resolved_path):
        raise _error(field_name, f"file does not exist: {resolved_path}")
    return candidate_path


def validate_session_config(config, require_model_file=False, require_label_file=False):
    if not isinstance(config, dict):
        raise _error("session_config", "must contain a JSON object")
    normalized = deepcopy(DEFAULT_SESSION_CONFIG)
    normalized.update(config)
    normalized["schema_version"] = SCHEMA_VERSION

    if normalized["visualizeFlag"] not in ("visualize", "robot"):
        raise _error("visualizeFlag", "invalid value (expected visualize or robot)")
    if normalized["useCPU"] not in ("CPU", "GPU"):
        raise _error("useCPU", "invalid value (expected CPU or GPU)")

    # EPD-8 keeps useCPU for backwards compatibility while execution_backend
    # becomes the explicit provider selection used by new deployments.
    if "execution_backend" not in config:
        normalized["execution_backend"] = (
            "cpu" if normalized["useCPU"] == "CPU" else "cuda"
        )
    else:
        normalized["execution_backend"] = normalize_execution_backend(
            normalized.get("execution_backend")
        )
    normalized["execution_backend_gpu_index"] = int(
        normalized.get("execution_backend_gpu_index", 0)
    )
    if normalized["execution_backend_gpu_index"] < 0:
        raise _error("execution_backend_gpu_index", "must be >= 0")

    normalized["intra_op_num_threads"] = int(normalized.get("intra_op_num_threads", 0))
    if normalized["intra_op_num_threads"] < 0:
        raise _error("intra_op_num_threads", "must be >= 0")

    normalized["image_transport"] = str(normalized.get("image_transport", "raw")).lower()
    if normalized["image_transport"] not in IMAGE_TRANSPORT_CHOICES:
        raise _error(
            "image_transport",
            "invalid value (expected one of " + ", ".join(IMAGE_TRANSPORT_CHOICES) + ")",
        )

    normalized["publish_detection_segmentation"] = bool(
        normalized.get("publish_detection_segmentation", True)
    )
    normalized["confidence_threshold"] = float(normalized.get("confidence_threshold", 0.5))
    if not 0.0 <= normalized["confidence_threshold"] <= 1.0:
        raise _error("confidence_threshold", "must be between 0.0 and 1.0")

    normalized["max_detections"] = int(normalized.get("max_detections", 100))
    if normalized["max_detections"] <= 0:
        raise _error("max_detections", "must be > 0")

    if require_model_file:
        normalized["path_to_model"] = validate_existing_file_path(
            normalized.get("path_to_model"), "path_to_model"
        )
    elif not str(normalized.get("path_to_model", "")).strip():
        raise _error("path_to_model", "path cannot be empty")

    if require_label_file:
        normalized["path_to_label_list"] = validate_existing_file_path(
            normalized.get("path_to_label_list"), "path_to_label_list"
        )
    elif not str(normalized.get("path_to_label_list", "")).strip():
        raise _error("path_to_label_list", "path cannot be empty")

    return normalized


def validate_usecase_mode(mode):
    mode_int = int(mode)
    if mode_int not in USECASE_MODE_CHOICES:
        raise _error("usecase_mode", "invalid value (expected integer in range 0-4)")
    return mode_int


def validate_usecase_config(config, require_mode_specific=False):
    if not isinstance(config, dict):
        raise _error("usecase_config", "must contain a JSON object")
    normalized = deepcopy(DEFAULT_USECASE_CONFIG)
    normalized.update(config)
    normalized["schema_version"] = SCHEMA_VERSION
    normalized["usecase_mode"] = validate_usecase_mode(normalized.get("usecase_mode", 0))

    mode = normalized["usecase_mode"]
    if mode == 1:
        if require_mode_specific or "class_list" in normalized:
            normalized["class_list"] = normalize_class_list(normalized.get("class_list", []))
    elif mode == 2:
        if require_mode_specific:
            normalized["path_to_color_template"] = validate_existing_file_path(
                normalized.get("path_to_color_template"), "path_to_color_template"
            )
        elif "path_to_color_template" in normalized:
            path_value = str(normalized.get("path_to_color_template") or "").strip()
            if not path_value:
                raise _error("path_to_color_template", "path cannot be empty")
            normalized["path_to_color_template"] = path_value

        normalized["color_match_histogram_metric"] = normalize_color_histogram_metric(
            normalized.get("color_match_histogram_metric", "Correlation")
        )
    elif mode == 4:
        if require_mode_specific or "track_type" in normalized:
            normalized["track_type"] = normalize_track_type(normalized.get("track_type", ""))

    return normalized


def validate_input_topic_config(config):
    if not isinstance(config, dict):
        raise _error("input_image_topic_config", "must contain a JSON object")
    normalized = deepcopy(DEFAULT_INPUT_TOPIC_CONFIG)
    normalized.update(config)
    normalized["schema_version"] = SCHEMA_VERSION

    topic = str(normalized.get("input_image_topic") or "").strip()
    if not topic:
        raise _error("input_image_topic", "must not be empty")
    normalized["input_image_topic"] = topic
    return normalized


def migrate_session_config(config):
    return validate_session_config(config)


def migrate_usecase_config(config):
    return validate_usecase_config(config)


def migrate_input_topic_config(config):
    return validate_input_topic_config(config)
