"""EPD-1 integration for the Camera Assistant."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from windows.camera_assistant import CameraAssistantWindow


def apply_epd1_productization(main_window):
    """Attach the EPD-1 Camera Assistant to the refreshed Deploy workflow."""
    if getattr(main_window, "_epd1_productization_applied", False):
        return
    main_window._epd1_productization_applied = True

    assistant = CameraAssistantWindow(main_window.deploy_window, parent=main_window)
    assistant.setWindowFlag(Qt.Window, True)
    main_window.camera_assistant = assistant

    _add_camera_assistant_control(main_window)
    _install_shortcut(main_window)
    assistant.health_updated.connect(
        lambda result: _apply_health_to_deploy(main_window, result)
    )


def _add_camera_assistant_control(main_window):
    deploy = main_window.deploy_window
    camera_card = _find_section_card(deploy, "CAMERA INPUT")
    if camera_card is None or camera_card.layout() is None:
        return

    row = QHBoxLayout()
    row.setContentsMargins(0, 1, 0, 0)
    row.setSpacing(8)

    health_label = QLabel("Camera health: not checked", camera_card)
    health_label.setObjectName("cameraAssistantSummary")
    health_label.setToolTip(
        "EPD-1 checks ROS 2, RGB, aligned depth, CameraInfo, rate and resolution."
    )
    row.addWidget(health_label, 1)

    button = QPushButton("Camera Assistant", camera_card)
    button.setObjectName("cameraAssistantButton")
    button.setCursor(Qt.PointingHandCursor)
    button.setMinimumHeight(34)
    button.setToolTip(
        "Open camera diagnostics for RGB, aligned depth, CameraInfo and ROS 2 health."
    )
    button.clicked.connect(lambda: _show_camera_assistant(main_window))
    row.addWidget(button, 0, Qt.AlignRight)
    camera_card.layout().addLayout(row)

    deploy._epd1_camera_assistant_button = button
    deploy._epd1_camera_health_label = health_label
    deploy.setStyleSheet(
        deploy.styleSheet()
        + """
        QPushButton#cameraAssistantButton {
            color: #d3dceb;
            background-color: #1c2531;
            border: 1px solid #40506a;
            border-radius: 8px;
            padding: 5px 12px;
            font-size: 10px;
            font-weight: 700;
        }
        QPushButton#cameraAssistantButton:hover {
            color: #ffffff;
            background-color: #263246;
            border-color: #657a9e;
        }
        QLabel#cameraAssistantSummary {
            color: #8f9db1;
            font-size: 10px;
        }
        QLabel#cameraAssistantSummary[healthState="ready"] {
            color: #9fd7a9;
        }
        QLabel#cameraAssistantSummary[healthState="partial"] {
            color: #dfcf92;
        }
        QLabel#cameraAssistantSummary[healthState="missing"] {
            color: #df9a9a;
        }
        """
    )


def _find_section_card(deploy, section_title):
    for label in deploy.findChildren(QLabel):
        if label.text().strip() != section_title:
            continue
        parent = label.parentWidget()
        if isinstance(parent, QFrame) and parent.objectName() == "sectionCard":
            return parent
    return None


def _install_shortcut(main_window):
    shortcut = QShortcut(QKeySequence("Ctrl+Shift+C"), main_window.deploy_window)
    shortcut.activated.connect(lambda: _show_camera_assistant(main_window))
    main_window._epd1_camera_shortcut = shortcut


def _show_camera_assistant(main_window):
    assistant = main_window.camera_assistant
    assistant.show()
    assistant.raise_()
    assistant.activateWindow()
    assistant.refresh_health()


def _apply_health_to_deploy(main_window, result):
    deploy = main_window.deploy_window
    streams = result.get("streams", {})
    rgb = streams.get("rgb", {})
    depth = streams.get("depth", {})
    info = streams.get("camerainfo", {})
    connected = bool(result.get("ros_connected"))
    usecase_mode = int(result.get("usecase_mode", 0))
    requires_3d = usecase_mode in (3, 4)

    if connected:
        deploy._image_topics_cache = list(result.get("image_topics", []))
        deploy._image_topics_cache_ts = time.time()
        selected = str(result.get("selected_rgb") or "").strip()
        if selected and selected in deploy._image_topics_cache:
            deploy._epd0_topic_discovery_state = "detected"
            deploy._epd0_topic_discovery_error = ""

    rgb_live = rgb.get("state") == "live"
    depth_live = depth.get("state") == "live"
    info_live = info.get("state") == "live"

    if connected and rgb_live and (not requires_3d or (depth_live and info_live)):
        state = "ready"
    elif connected and rgb_live:
        state = "partial"
    else:
        state = "missing"

    label = getattr(deploy, "_epd1_camera_health_label", None)
    if label is not None:
        label.setText(_summary_text(connected, rgb_live, depth_live, info_live, requires_3d))
        label.setProperty("healthState", state)
        label.style().unpolish(label)
        label.style().polish(label)

    controller = getattr(main_window, "_deploy_ui_controller", None)
    if controller is not None:
        controller.sync()


def _summary_text(connected, rgb_live, depth_live, info_live, requires_3d):
    if not connected:
        return "Camera health: ROS 2 unavailable"

    rgb_text = "RGB ✓" if rgb_live else "RGB ✕"
    if not requires_3d:
        return f"Camera health: {rgb_text}  •  depth/info optional for this mode"

    depth_text = "Depth ✓" if depth_live else "Depth ✕"
    info_text = "CameraInfo ✓" if info_live else "CameraInfo ✕"
    return f"Camera health: {rgb_text}  •  {depth_text}  •  {info_text}"
