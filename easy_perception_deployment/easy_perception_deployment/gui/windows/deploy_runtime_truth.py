"""Deploy runtime-truth fixes shared by the refreshed EPD GUI.

This module keeps two pieces of UI state aligned with the runtime that EPD can
actually use:

* a locally compiled CPU provider is a valid CPU backend even when the optional
  Docker CPU image is absent;
* EPD-owned image topics are outputs/internal transport and must never be
  offered or accepted as the upstream camera source.

It is intentionally narrow and does not change inference, camera, or task
semantics.
"""

from __future__ import annotations

from types import MethodType

from windows import backend_manager


_EPD_INTERNAL_TOPIC_PREFIX = "/easy_perception_deployment/"
_DEFAULT_CAMERA_TOPIC = "/camera/camera/color/image_raw"


def is_camera_source_topic(topic):
    """Return True only for non-empty image topics that are upstream of EPD."""
    value = str(topic or "").strip()
    return bool(value) and not value.startswith(_EPD_INTERNAL_TOPIC_PREFIX)


def filter_camera_source_topics(topics):
    """Remove EPD outputs/internal ingress topics from camera discovery results."""
    filtered = []
    seen = set()
    for topic in topics or []:
        value = str(topic or "").strip()
        if not is_camera_source_topic(value) or value in seen:
            continue
        filtered.append(value)
        seen.add(value)
    return filtered


def augment_local_cpu_readiness(probe):
    """Treat a verified local CPU provider as ready without requiring Docker.

    Docker image readiness remains in the probe and Performance dialog as a
    separate deployment-path check. This only corrects the compact execution
    backend truth: a local ROS 2 build that reports ``cpu: true`` through
    ``epd_backend_probe`` can execute EPD on CPU.
    """
    result = dict(probe or {})
    ready = dict(result.get("ready") or {})
    compiled = result.get("compiled")
    local_cpu_ready = bool(
        isinstance(compiled, dict) and compiled.get("cpu", True)
    )

    if local_cpu_ready:
        ready["cpu"] = True

    result["ready"] = ready
    result["local_provider_ready"] = {"cpu": local_cpu_ready}

    # Keep an already verified CUDA recommendation. Otherwise CPU is the safe
    # local/reference path when it is compiled and available.
    if not ready.get("cuda", False) and ready.get("cpu", False):
        result["recommended"] = "cpu"
    return result


def _install_backend_probe_truth():
    current = backend_manager.probe_environment
    if getattr(current, "_epd_local_cpu_truth", False):
        return

    original = current

    def local_aware_probe():
        return augment_local_cpu_readiness(original())

    local_aware_probe._epd_local_cpu_truth = True
    local_aware_probe._epd_original_probe = original
    backend_manager.probe_environment = local_aware_probe


def _default_camera_topic(window):
    configured_default = str(
        getattr(window, "DEFAULT_INPUT_TOPIC", _DEFAULT_CAMERA_TOPIC) or ""
    ).strip()
    if is_camera_source_topic(configured_default):
        return configured_default
    return _DEFAULT_CAMERA_TOPIC


def _repair_invalid_saved_topic(window):
    configured = str(getattr(window, "_input_image_topic", "") or "").strip()
    if not configured or is_camera_source_topic(configured):
        return configured

    fallback = _default_camera_topic(window)
    logger = getattr(window, "deploy_logger", None)
    if logger is not None:
        logger.warning(
            "EPD-owned topic '%s' cannot be used as camera input; restoring '%s'.",
            configured,
            fallback,
        )

    combo = window.topic_button
    combo.blockSignals(True)
    if combo.findText(fallback) < 0:
        combo.addItem(fallback)
    combo.setCurrentText(fallback)
    combo.blockSignals(False)

    # Persist through DeployWindow's existing validated configuration path.
    window._input_image_topic = fallback
    window.setImageInput()
    return fallback


def _set_property(widget, name, value):
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def _show_invalid_input_truth(main_window, controller):
    window = main_window.deploy_window
    configured = str(window.topic_button.currentText() or "").strip()
    if not configured:
        configured = str(getattr(window, "_input_image_topic", "") or "").strip()
    if not configured.startswith(_EPD_INTERNAL_TOPIC_PREFIX):
        return

    controller.topic_value.setText(configured)
    controller.topic_value.setToolTip(configured)
    controller.topic_state.setText("✕ Invalid")
    _set_property(controller.topic_state, "state", "blocked")
    # Reuse the existing red missing/invalid visual language from EPD-0.
    _set_property(controller.topic_state, "epd0State", "missing")

    message = (
        "Run disabled: EPD output/internal topic cannot be used as camera input. "
        f"Choose an upstream RGB topic such as {_default_camera_topic(window)}."
    )
    window.run_button.setEnabled(False)
    window.run_button.setToolTip(message)
    window.validation_label.setText(message)

    if controller.header_badge.text() != "RUNNING":
        controller.header_badge.setText("SETUP REQUIRED")
        _set_property(controller.header_badge, "state", "blocked")
        _set_property(controller.header_badge, "epd0State", "")


def _install_camera_topic_truth(main_window):
    window = main_window.deploy_window
    controller = main_window._deploy_ui_controller

    _repair_invalid_saved_topic(window)
    window._image_topics_cache = filter_camera_source_topics(
        getattr(window, "_image_topics_cache", [])
    )

    # EPD-0's worker callback is looked up dynamically by its signal lambda, so
    # replacing the bound method here also filters future Refresh topics scans.
    original_success = getattr(window, "_epd0_topic_success", None)
    if original_success is not None and not getattr(
        original_success, "_epd_camera_source_truth", False
    ):
        def safe_topic_success(self, topics, current):
            safe_topics = filter_camera_source_topics(topics)
            safe_current = str(current or "").strip()
            if not is_camera_source_topic(safe_current):
                safe_current = _default_camera_topic(self)
            return original_success(safe_topics, safe_current)

        safe_topic_success._epd_camera_source_truth = True
        window._epd0_topic_success = MethodType(safe_topic_success, window)

    # Acceptance stability is installed before EPD-0 (see main.py), so EPD-0
    # remains the owner of detected/configured camera chips. Add only a final
    # defensive check for manually typed internal EPD topics.
    if not getattr(controller, "_deploy_runtime_truth_wrapped", False):
        original_sync = controller.sync

        def runtime_truth_sync(self):
            original_sync()
            _show_invalid_input_truth(main_window, self)

        controller.sync = MethodType(runtime_truth_sync, controller)
        controller._deploy_runtime_truth_wrapped = True
        timer = getattr(controller, "_summary_timer", None)
        if timer is not None:
            try:
                timer.timeout.disconnect()
            except (RuntimeError, TypeError):
                pass
            timer.timeout.connect(controller.sync)

    controller.sync()


def apply_deploy_runtime_truth(main_window):
    """Install local-backend and camera-source truth corrections once."""
    if getattr(main_window, "_deploy_runtime_truth_applied", False):
        return
    main_window._deploy_runtime_truth_applied = True
    _install_backend_probe_truth()
    _install_camera_topic_truth(main_window)
