"""EPD-0 productization: camera truth, clearer controls and in-app help."""

from __future__ import annotations

import time
from types import MethodType

from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QFrame, QLabel, QPushButton

from windows.Deploy import ImageTopicsWorker
from windows.help_window import HelpWindow


_IMAGE_TOPIC_TIMEOUT_SEC = 6


def apply_epd0_productization(main_window):
    """Apply EPD-0 UX improvements to an already-constructed MainWindow."""
    if getattr(main_window, "_epd0_productization_applied", False):
        return
    main_window._epd0_productization_applied = True

    main_window.help_window = HelpWindow(main_window)
    _install_launcher_help(main_window)
    _install_deploy_productization(main_window)
    _install_f1_shortcuts(main_window)


def _show_help(main_window, topic="Quick Start"):
    main_window.help_window.select_topic(topic)
    main_window.help_window.show()
    main_window.help_window.raise_()
    main_window.help_window.activateWindow()


def _install_launcher_help(main_window):
    root = main_window.layout()
    if root is None:
        return

    button = QPushButton("Help & Guides   F1", main_window)
    button.setObjectName("epd0HelpButton")
    button.setCursor(Qt.PointingHandCursor)
    button.setMinimumHeight(36)
    button.setToolTip(
        "Open camera, model, training, deployment and troubleshooting guidance."
    )
    button.clicked.connect(lambda: _show_help(main_window))

    insert_index = max(0, root.count() - 2)
    root.insertWidget(insert_index, button, 0, Qt.AlignRight)
    main_window.help_button = button

    main_window.setStyleSheet(
        main_window.styleSheet()
        + """
        QPushButton#epd0HelpButton {
            color: #c5cfdd;
            background-color: #171d26;
            border: 1px solid #334052;
            border-radius: 9px;
            padding: 6px 14px;
            font-size: 11px;
            font-weight: 600;
        }
        QPushButton#epd0HelpButton:hover {
            color: #ffffff;
            background-color: #202936;
            border-color: #60728c;
        }
        """
    )


def _install_f1_shortcuts(main_window):
    shortcuts = []
    for widget in (main_window, main_window.train_window, main_window.deploy_window):
        shortcut = QShortcut(QKeySequence("F1"), widget)
        shortcut.activated.connect(lambda mw=main_window: _show_help(mw))
        shortcuts.append(shortcut)
    main_window._epd0_help_shortcuts = shortcuts


def _install_deploy_productization(main_window):
    deploy = main_window.deploy_window
    controller = main_window._deploy_ui_controller

    deploy._epd0_topic_discovery_state = "checking"
    deploy._epd0_topic_discovery_error = ""

    _configure_deploy_tooltips(deploy)
    _preserve_configured_topic(deploy)
    _replace_topic_refresh(deploy)
    _add_deploy_help_button(main_window, deploy)
    _improve_section_copy(deploy)
    _wrap_summary_sync(main_window, controller)
    _append_deploy_style(deploy)
    controller.sync()


def _configure_deploy_tooltips(w):
    w.topic_button.setToolTip(
        "RGB image topic processed by EPD. RealSense default: "
        "/camera/camera/color/image_raw. Manual entry remains available."
    )
    w.refresh_topics_button.setToolTip(
        "Scan ROS 2 image topics. The saved topic is preserved if discovery fails."
    )
    w.visualize_button.setToolTip(
        "Detection overlay for human inspection. Off reduces visualization overhead; "
        "ROS perception results still publish."
    )
    w.segmentation_button.setToolTip(
        "Publish object-mask/segmentation output where supported. Useful for Mask R-CNN."
    )
    w.usecase_config_button.setToolTip(
        "Choose Classification, Counting, Color-Matching, Localization or Tracking."
    )
    w.transport_combo.setToolTip(
        "raw is simplest; compressed can reduce bandwidth but adds codec work."
    )
    w.docker_button.setToolTip(
        "CPU is the safe default. Use GPU only with a configured accelerated runtime."
    )
    w.confidence_spinbox.setToolTip(
        "Minimum accepted detection confidence. 0.50 is a reasonable starting point."
    )
    w.max_detections_spinbox.setToolTip(
        "Maximum detections processed per frame. Use a cap for predictable runtime."
    )
    w.use_defaults_button.setToolTip(
        "Restore bundled model, labels and the default RealSense RGB topic."
    )


def _preserve_configured_topic(w):
    configured = str(getattr(w, "_input_image_topic", "") or "").strip()
    if not configured:
        return ""

    current = w.topic_button.currentText().strip()
    if current:
        return current

    w.topic_button.blockSignals(True)
    if w.topic_button.findText(configured) < 0:
        w.topic_button.addItem(configured)
    w.topic_button.setCurrentText(configured)
    w.topic_button.blockSignals(False)
    return configured


def _replace_topic_refresh(w):
    original_refresh = w.refreshImageTopics

    def refresh_image_topics(self, select_topic=None):
        current = self.topic_button.currentText().strip()
        if isinstance(select_topic, str):
            current = select_topic.strip()
        current = current or str(self._input_image_topic or "").strip()

        self._epd0_topic_discovery_state = "checking"
        self._epd0_topic_discovery_error = ""
        _preserve_configured_topic(self)

        now = time.time()
        cache_age = now - self._image_topics_cache_ts
        if cache_age <= self._image_topics_cache_ttl_sec:
            self._apply_image_topics(self._image_topics_cache, current)
            _update_state_from_cache(self, current)
            return

        thread = self._topics_worker_thread
        if thread is not None and thread.isRunning():
            return

        self.refresh_topics_button.setEnabled(False)
        self.validation_label.setText("Checking ROS 2 image topics…")
        self._topics_worker = ImageTopicsWorker(
            timeout_sec=_IMAGE_TOPIC_TIMEOUT_SEC
        )
        self._topics_worker_thread = QThread(self)
        self._topics_worker.moveToThread(self._topics_worker_thread)

        self._topics_worker_thread.started.connect(self._topics_worker.run)
        self._topics_worker.signals.success.connect(
            lambda topics, selected=current: self._epd0_topic_success(
                topics,
                selected,
            )
        )
        self._topics_worker.signals.error.connect(self._epd0_topic_error)
        self._topics_worker.signals.finished.connect(
            self._on_topics_refresh_finished
        )
        self._topics_worker.signals.finished.connect(
            self._topics_worker_thread.quit
        )
        self._topics_worker_thread.finished.connect(
            self._topics_worker.deleteLater
        )
        self._topics_worker_thread.finished.connect(
            self._topics_worker_thread.deleteLater
        )
        self._topics_worker_thread.finished.connect(
            self._clear_topics_worker_refs
        )
        self._topics_worker_thread.start()

    def on_success(self, topics, current):
        self._image_topics_cache = topics
        self._image_topics_cache_ts = time.time()
        self._epd0_topic_discovery_error = ""
        self._apply_image_topics(topics, current)
        _update_state_from_cache(self, current)

    def on_error(self, message):
        self.deploy_logger.warning(message)
        self._epd0_topic_discovery_state = "unverified"
        self._epd0_topic_discovery_error = message
        configured = _preserve_configured_topic(self)
        if configured:
            self.validation_label.setText(
                "Camera discovery unavailable. Saved topic preserved: "
                f"{configured}. Check ROS 2/camera and Refresh topics."
            )
        else:
            self.validation_label.setText(
                "Camera discovery unavailable. Check ROS 2/camera or type a topic."
            )

    w.refreshImageTopics = MethodType(refresh_image_topics, w)
    w._epd0_topic_success = MethodType(on_success, w)
    w._epd0_topic_error = MethodType(on_error, w)

    try:
        w.refresh_topics_button.clicked.disconnect(original_refresh)
    except (RuntimeError, TypeError):
        pass
    w.refresh_topics_button.clicked.connect(w.refreshImageTopics)


def _update_state_from_cache(w, configured):
    topics = list(getattr(w, "_image_topics_cache", []) or [])
    if configured and configured in topics:
        w._epd0_topic_discovery_state = "detected"
    elif configured:
        w._epd0_topic_discovery_state = "configured_not_detected"
    else:
        w._epd0_topic_discovery_state = "missing"


def _add_deploy_help_button(main_window, deploy):
    header = deploy.findChild(QFrame, "deployHeader")
    if header is None or header.layout() is None:
        return

    button = QPushButton("?  Help", header)
    button.setObjectName("deployHelpButton")
    button.setCursor(Qt.PointingHandCursor)
    button.setToolTip("Open deployment and camera guidance (F1).")
    button.clicked.connect(
        lambda: _show_help(main_window, "Deploy Step-by-Step")
    )
    layout = header.layout()
    layout.insertWidget(max(0, layout.count() - 1), button, 0, Qt.AlignTop)
    deploy._epd0_help_button = button


def _improve_section_copy(deploy):
    replacements = {
        "Select the use case and runtime output options.": (
            "Choose the perception task and optional human/mask outputs. "
            "Hover controls for guidance."
        ),
        "Select an image topic from ROS 2 or type one manually.": (
            "Choose the RGB stream. RealSense default: "
            "/camera/camera/color/image_raw. Saved topics survive discovery failure."
        ),
        "Run stays blocked until required inputs are valid.": (
            "File checks stay strict. Camera state separates configured input "
            "from a topic detected live on ROS 2."
        ),
    }
    for label in deploy.findChildren(QLabel):
        text = label.text().strip()
        if text in replacements:
            label.setText(replacements[text])


def _wrap_summary_sync(main_window, controller):
    if getattr(controller, "_epd0_sync_wrapped", False):
        return
    controller._epd0_sync_wrapped = True
    controller._epd0_original_sync = controller.sync

    def sync(self):
        self._epd0_original_sync()
        _apply_deploy_truth(main_window, self)
        main_window._keep_deploy_actions_neutral()

    controller.sync = MethodType(sync, controller)
    try:
        controller._summary_timer.timeout.disconnect()
    except (RuntimeError, TypeError):
        pass
    controller._summary_timer.timeout.connect(controller.sync)


def _apply_deploy_truth(main_window, controller):
    w = main_window.deploy_window
    configured = _preserve_configured_topic(w)
    validation = w.validation_label.text().strip().lower()

    discovery_failed = (
        "timed out" in validation
        or "unable to refresh topics" in validation
        or "unable to refresh topics from ros2" in validation
    )
    if discovery_failed:
        w._epd0_topic_discovery_state = "unverified"
        w._epd0_topic_discovery_error = validation

    topics = list(getattr(w, "_image_topics_cache", []) or [])
    cache_ts = float(getattr(w, "_image_topics_cache_ts", 0.0) or 0.0)
    state = getattr(w, "_epd0_topic_discovery_state", "checking")

    if configured and configured in topics:
        state = "detected"
    elif getattr(w, "_epd0_topic_discovery_error", ""):
        state = "unverified"
    elif cache_ts > 0 and configured:
        state = "configured_not_detected"
    elif configured and state not in ("detected", "unverified"):
        state = "checking"
    elif not configured:
        state = "missing"
    w._epd0_topic_discovery_state = state

    w.visualize_button.setText(
        "Detection overlay  •  On"
        if getattr(w, "visualizeFlag", False)
        else "Detection overlay  •  Off"
    )
    w.segmentation_button.setText(
        "Object masks  •  On"
        if getattr(w, "publish_detection_segmentation", False)
        else "Object masks  •  Off"
    )

    controller.topic_value.setText(configured or "Not configured")
    controller.topic_value.setToolTip(configured)
    _set_camera_chip(controller.topic_state, state, configured)

    model_ready = "✅" in w.model_readiness_label.text()
    labels_ready = "✅" in w.label_list_readiness_label.text()
    running = controller.header_badge.text() == "RUNNING"

    if not running and model_ready and labels_ready and configured:
        if state != "detected":
            controller.header_badge.setText("CONFIGURED")
            _set_property(controller.header_badge, "epd0State", "configured")
        else:
            _set_property(controller.header_badge, "epd0State", "")
    else:
        _set_property(controller.header_badge, "epd0State", "")

    if model_ready and labels_ready:
        _set_camera_message(w, state, configured)


def _set_camera_chip(chip, state, configured):
    if state == "detected":
        chip.setText("✓ Detected")
        _set_property(chip, "epd0State", "detected")
    elif configured:
        chip.setText("⚠ Configured")
        _set_property(chip, "epd0State", "configured")
    else:
        chip.setText("✕ Missing")
        _set_property(chip, "epd0State", "missing")


def _set_camera_message(w, state, configured):
    if state == "unverified" and configured:
        w.validation_label.setText(
            "Camera discovery unavailable; saved topic preserved: "
            f"{configured}. Live camera presence is unverified."
        )
    elif state == "configured_not_detected" and configured:
        w.validation_label.setText(
            "Saved camera topic is configured but not detected in the latest scan: "
            f"{configured}. Start the camera or Refresh topics."
        )
    elif state == "detected":
        w.validation_label.setText(
            "Camera input detected on ROS 2: "
            f"{configured}. Required deployment inputs are ready."
        )
    elif state == "missing":
        w.validation_label.setText(
            "Camera input is not configured. Refresh topics or type one manually."
        )


def _set_property(widget, name, value):
    widget.setProperty(name, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def _append_deploy_style(deploy):
    deploy.setStyleSheet(
        deploy.styleSheet()
        + """
        QPushButton#deployHelpButton {
            color: #bfcadd;
            background-color: transparent;
            border: 1px solid #354256;
            border-radius: 8px;
            padding: 5px 12px;
            font-size: 10px;
            font-weight: 600;
        }
        QPushButton#deployHelpButton:hover {
            color: #ffffff;
            background-color: #1e2632;
            border-color: #5c6f8c;
        }
        QLabel#summaryState[epd0State="detected"] {
            color: #a8e6b0;
            background-color: #12351d;
            border: 1px solid #286d38;
        }
        QLabel#summaryState[epd0State="configured"] {
            color: #e8d89f;
            background-color: #302a16;
            border: 1px solid #6a5a24;
        }
        QLabel#summaryState[epd0State="missing"] {
            color: #e7a6a6;
            background-color: #34191b;
            border: 1px solid #713139;
        }
        QLabel#headerBadge[epd0State="configured"] {
            color: #eadca9;
            background-color: #332c17;
            border: 1px solid #6b5a25;
        }
        """
    )
