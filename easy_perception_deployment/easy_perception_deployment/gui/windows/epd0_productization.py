"""EPD-0 productization layer: camera truth, clearer controls and in-app help.

This module intentionally sits on top of the existing Train/Deploy windows so EPD-0 can
improve operator clarity without changing inference, ROS message, training or launch logic.
"""

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
    _install_deploy_help(main_window)
    _install_f1_shortcuts(main_window)


def _show_help(main_window, topic=None):
    help_window = main_window.help_window
    if topic:
        help_window.select_topic(topic)
    help_window.show()
    help_window.raise_()
    help_window.activateWindow()


def _install_launcher_help(main_window):
    root = main_window.layout()
    if root is None:
        return

    button = QPushButton("Help & Guides   F1", main_window)
    button.setObjectName("epd0HelpButton")
    button.setCursor(Qt.PointingHandCursor)
    button.setMinimumHeight(36)
    button.setToolTip(
        "Open the offline EPD guide: camera setup, model choice, training, deployment, "
        "perception modes and troubleshooting."
    )
    button.clicked.connect(lambda: _show_help(main_window, "Quick Start"))

    # MainWindow layout: header, section label, workflow grid, stretch, footer.
    # Place help directly below the workflow cards without restructuring the launcher.
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
        shortcut.activated.connect(lambda mw=main_window: _show_help(mw, "Quick Start"))
        shortcuts.append(shortcut)
    main_window._epd0_help_shortcuts = shortcuts


def _install_deploy_help(main_window):
    deploy = main_window.deploy_window
    controller = main_window._deploy_ui_controller

    deploy._epd0_topic_discovery_state = "checking"
    deploy._epd0_topic_discovery_error = ""

    _configure_deploy_tooltips(deploy)
    _preserve_configured_topic(deploy)
    _replace_topic_refresh(deploy)
    _add_deploy_help_button(main_window, deploy)
    _improve_section_copy(deploy)
    _wrap_deploy_summary_sync(main_window, controller)

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

    # The initial topic scan starts inside DeployWindow.__init__ before EPD-0 is
    # installed. If it later times out, the wrapped summary sync below notices the
    # original warning and still preserves the configured topic truthfully.
    controller.sync()


def _configure_deploy_tooltips(w):
    w.topic_button.setToolTip(
        "RGB image topic processed by EPD. RealSense D435i default: "
        "/camera/camera/color/image_raw. You can type a topic manually even when "
        "ROS topic discovery is unavailable."
    )
    w.refresh_topics_button.setToolTip(
        "Scan the ROS 2 graph for sensor_msgs/msg/Image topics. The saved topic is "
        "preserved if discovery times out or the camera is not running yet."
    )
    w.visualize_button.setToolTip(
        "Detection overlay: draw EPD visualization output for human inspection. "
        "Turning this off reduces visualization overhead; ROS perception results still publish."
    )
    w.segmentation_button.setToolTip(
        "Object masks: publish segmentation-related per-object output where supported. "
        "Useful for Mask R-CNN and manipulation; it can increase processing and bandwidth."
    )
    w.usecase_config_button.setToolTip(
        "Choose what EPD should do: Classification, Counting, Color-Matching, "
        "Localization (3D geometry), or Tracking (3D geometry plus persistent IDs)."
    )
    w.transport_combo.setToolTip(
        "ROS image transport. raw is simplest and lowest CPU overhead; compressed can reduce "
        "network bandwidth at the cost of encode/decode work."
    )
    w.docker_button.setToolTip(
        "Inference device. CPU is the safest default. Use GPU only when the deployment "
        "environment and model runtime are configured for GPU acceleration."
    )
    w.confidence_spinbox.setToolTip(
        "Minimum confidence accepted as a detection. Higher values reduce false positives "
        "but can miss difficult objects. 0.50 is a reasonable starting point."
    )
    w.max_detections_spinbox.setToolTip(
        "Maximum detections processed per frame. Lower limits can protect runtime performance "
        "in crowded scenes."
    )
    w.use_defaults_button.setToolTip(
        "Restore the bundled model, labels and default RealSense RGB topic."
    )
    w.run_button.setToolTip(
        "Start perception with the configured model, labels, camera topic and mode."
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
    """Use a slightly more tolerant scan and explicitly retain configured-topic truth."""
    original_refresh = w.refreshImageTopics

    def refresh_image_topics(self, select_topic=None):
        current_topic = (
            select_topic
            if select_topic is not None
            else self.topic_button.currentText().strip()
        )
        current_topic = current_topic or str(getattr(self, "_input_image_topic", "") or "").strip()
        self._epd0_topic_discovery_state = "checking"
        self._epd0_topic_discovery_error = ""
        _preserve_configured_topic(self)

        now = time.time()
        if (now - self._image_topics_cache_ts) <= self._image_topics_cache_ttl_sec:
            self._apply_image_topics(self._image_topics_cache, current_topic)
            _update_discovery_state_from_cache(self, current_topic)
            return

        if self._topics_worker_thread is not None and self._topics_worker_thread.isRunning():
            return

        self.refresh_topics_button.setEnabled(False)
        self.validation_label.setText("Checking ROS 2 image topics…")

        self._topics_worker = ImageTopicsWorker(timeout_sec=_IMAGE_TOPIC_TIMEOUT_SEC)
        self._topics_worker_thread = QThread(self)
        self._topics_worker.moveToThread(self._topics_worker_thread)

        self._topics_worker_thread.started.connect(self._topics_worker.run)
        self._topics_worker.signals.success.connect(
            lambda topics, selected=current_topic:
            self._epd0_on_topics_refresh_success(topics, selected)
        )
        self._topics_worker.signals.error.connect(self._epd0_on_topics_refresh_error)
        self._topics_worker.signals.finished.connect(self._on_topics_refresh_finished)
        self._topics_worker.signals.finished.connect(self._topics_worker_thread.quit)
        self._topics_worker_thread.finished.connect(self._topics_worker.deleteLater)
        self._topics_worker_thread.finished.connect(self._topics_worker_thread.deleteLater)
        self._topics_worker_thread.finished.connect(self._clear_topics_worker_refs)
        self._topics_worker_thread.start()

    def on_success(self, topics, current_topic):
        self._image_topics_cache = topics
        self._image_topics_cache_ts = time.time()
        self._epd0_topic_discovery_error = ""
        self._apply_image_topics(topics, current_topic)
        _update_discovery_state_from_cache(self, current_topic)

    def on_error(self, message):
        self.deploy_logger.warning(message)
        self._epd0_topic_discovery_state = "unverified"
        self._epd0_topic_discovery_error = message
        configured = _preserve_configured_topic(self)
        if configured:
            self.validation_label.setText(
                "Camera discovery unavailable. Saved topic preserved: "
                f"{configured}. Check ROS 2/camera and click Refresh topics."
            )
        else:
            self.validation_label.setText(
                "Camera discovery unavailable. Check ROS 2/camera, then click Refresh topics "
                "or type an image topic manually."
            )

    w.refreshImageTopics = MethodType(refresh_image_topics, w)
    w._epd0_on_topics_refresh_success = MethodType(on_success, w)
    w._epd0_on_topics_refresh_error = MethodType(on_error, w)

    # Preserve other clicked hooks (for example the refreshed UI summary hook) and
    # replace only the original refresh slot.
    try:
        w.refresh_topics_button.clicked.disconnect(original_refresh)
    except (RuntimeError, TypeError):
        pass
    w.refresh_topics_button.clicked.connect(w.refreshImageTopics)


def _update_discovery_state_from_cache(w, configured_topic):
    configured_topic = str(configured_topic or "").strip()
    topics = list(getattr(w, "_image_topics_cache", []) or [])
    if configured_topic and configured_topic in topics:
        w._epd0_topic_discovery_state = "detected"
    elif configured_topic:
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
    button.setToolTip("Open deployment, camera and troubleshooting guidance (F1).")
    button.clicked.connect(lambda: _show_help(main_window, "Deploy Step-by-Step"))

    layout = header.layout()
    insert_index = max(0, layout.count() - 1)
    layout.insertWidget(insert_index, button, 0, Qt.AlignTop)
    deploy._epd0_help_button = button


def _improve_section_copy(deploy):
    for label in deploy.findChildren(QLabel):
        text = label.text().strip()
        if text == "Select the use case and runtime output options.":
            label.setText(
                "Choose the perception task and optional human/segmentation outputs. "
                "Hover controls for guidance."
            )
        elif text == "Select an image topic from ROS 2 or type one manually.":
            label.setText(
                "Choose the RGB image stream. RealSense default: "
                "/camera/camera/color/image_raw. Saved topics are kept if discovery fails."
            )
        elif text == "Run stays blocked until required inputs are valid.":
            label.setText(
                "Files must be valid. Camera state distinguishes configured input from a topic "
                "currently detected on ROS 2."
            )


def _wrap_deploy_summary_sync(main_window, controller):
    if getattr(controller, "_epd0_sync_wrapped", False):
        return
    controller._epd0_sync_wrapped = True
    controller._epd0_original_sync = controller.sync

    def sync(self):
        self._epd0_original_sync()
        _apply_deploy_truth(main_window, self)
        main_window._keep_deploy_actions_neutral()

    controller.sync = MethodType(sync, controller)

    # The timer was connected to the original bound method before EPD-0 existed.
    # Reconnect it to the wrapper; the wrapper also keeps legacy green/red surfaces neutral.
    try:
        controller._summary_timer.timeout.disconnect()
    except (RuntimeError, TypeError):
        pass
    controller._summary_timer.timeout.connect(controller.sync)


def _apply_deploy_truth(main_window, controller):
    w = main_window.deploy_window
    configured = _preserve_configured_topic(w)
    if not configured:
        configured = w.topic_button.currentText().strip()

    validation_text = w.validation_label.text().strip()
    if (
        "timed out" in validation_text.lower()
        or "unable to refresh topics" in validation_text.lower()
        or "unable to refresh topics from ros2" in validation_text.lower()
    ):
        w._epd0_topic_discovery_state = "unverified"
        w._epd0_topic_discovery_error = validation_text

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

    # Make legacy terminology operator-facing without changing backend semantics.
    w.visualize_button.setText(
        "Detection overlay  •  On" if getattr(w, "visualizeFlag", False)
        else "Detection overlay  •  Off"
    )
    w.segmentation_button.setText(
        "Object masks  •  On" if getattr(w, "publish_detection_segmentation", False)
        else "Object masks  •  Off"
    )

    controller.topic_value.setText(configured or "Not configured")
    controller.topic_value.setToolTip(configured)

    chip = controller.topic_state
    if state == "detected":
        chip.setText("✓ Detected")
        _set_dynamic_state(chip, "detected")
    elif state in ("unverified", "configured_not_detected", "checking") and configured:
        chip.setText("⚠ Configured")
        _set_dynamic_state(chip, "configured")
    else:
        chip.setText("✕ Missing")
        _set_dynamic_state(chip, "missing")

    model_ready = "✅" in w.model_readiness_label.text()
    labels_ready = "✅" in w.label_list_readiness_label.text()
    running = controller.header_badge.text() == "RUNNING"

    if not running and model_ready and labels_ready and configured and state != "detected":
        controller.header_badge.setText("CONFIGURED")
        controller.header_badge.setProperty("epd0State", "configured")
        controller.header_badge.style().unpolish(controller.header_badge)
        controller.header_badge.style().polish(controller.header_badge)
    else:
        controller.header_badge.setProperty("epd0State", "")

    # Keep model/label errors authoritative. Improve only camera-facing readiness text.
    if model_ready and labels_ready:
        if state == "unverified" and configured:
            w.validation_label.setText(
                "Camera discovery unavailable; saved topic preserved: "
                f"{configured}. Run may be configured, but live camera presence is unverified."
            )
        elif state == "configured_not_detected" and configured:
            w.validation_label.setText(
                "Saved camera topic is configured but was not detected in the latest ROS 2 scan: "
                f"{configured}. Start the camera, choose another topic, or refresh again."
            )
        elif state == "detected":
            w.validation_label.setText(
                f"Camera input detected on ROS 2: {configured}. Required deployment inputs are ready."
            )
        elif state == "missing":
            w.validation_label.setText(
                "Camera input is not configured. Start the camera and Refresh topics, or type an "
                "image topic manually."
            )


def _set_dynamic_state(widget, value):
    widget.setProperty("epd0State", value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
