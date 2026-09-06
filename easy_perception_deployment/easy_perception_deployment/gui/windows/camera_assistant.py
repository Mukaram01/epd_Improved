"""Camera health diagnostics for the EPD-1 Camera Assistant."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


DEFAULT_RGB_TOPIC = "/camera/camera/color/image_raw"
DEFAULT_DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
DEFAULT_CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"

IMAGE_TYPE = "sensor_msgs/msg/Image"
CAMERA_INFO_TYPE = "sensor_msgs/msg/CameraInfo"


@dataclass
class StreamHealth:
    """Health snapshot for one ROS camera stream."""

    name: str
    topic: str
    expected_type: str
    graph_present: bool = False
    sample_received: bool = False
    message_type: str = ""
    width: int | None = None
    height: int | None = None
    encoding: str = ""
    rate_hz: float | None = None
    stamp_age_ms: float | None = None
    aligned: bool | None = None
    detail: str = ""

    @property
    def state(self):
        if not self.topic:
            return "missing"
        if not self.graph_present:
            return "missing"
        if not self.sample_received:
            return "unresponsive"
        return "live"


def parse_topic_list(output):
    """Parse `ros2 topic list -t` into a topic -> message type mapping."""
    topics = {}
    for raw_line in str(output or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(\S+)\s+\[(.+)\]$", line)
        if not match:
            continue
        topic = match.group(1)
        type_text = match.group(2)
        topic_types = [part.strip() for part in type_text.split(",")]
        topics[topic] = topic_types[0] if topic_types else ""
    return topics


def infer_camera_topics(topic_types, selected_rgb):
    """Infer RGB/depth/CameraInfo targets without silently changing Deploy config."""
    selected_rgb = str(selected_rgb or "").strip()
    image_topics = sorted(
        topic for topic, msg_type in topic_types.items() if msg_type == IMAGE_TYPE
    )
    info_topics = sorted(
        topic for topic, msg_type in topic_types.items() if msg_type == CAMERA_INFO_TYPE
    )

    rgb_topic = selected_rgb or _choose_rgb_topic(image_topics)
    depth_topic = _choose_depth_topic(image_topics, rgb_topic)
    info_topic = _choose_camera_info_topic(info_topics, rgb_topic)

    return {
        "rgb": rgb_topic,
        "depth": depth_topic,
        "camera_info": info_topic,
        "image_topics": image_topics,
        "camera_info_topics": info_topics,
    }


def _choose_rgb_topic(image_topics):
    if DEFAULT_RGB_TOPIC in image_topics:
        return DEFAULT_RGB_TOPIC
    for topic in image_topics:
        lowered = topic.lower()
        if "color" in lowered and "depth" not in lowered:
            return topic
    for topic in image_topics:
        if "depth" not in topic.lower():
            return topic
    return ""


def _choose_depth_topic(image_topics, rgb_topic):
    derived = _derive_related_topic(
        rgb_topic,
        "/color/image_raw",
        "/aligned_depth_to_color/image_raw",
    )
    for candidate in (derived, DEFAULT_DEPTH_TOPIC):
        if candidate and candidate in image_topics:
            return candidate
    for topic in image_topics:
        if "aligned_depth" in topic.lower():
            return topic
    for topic in image_topics:
        if "depth" in topic.lower():
            return topic
    return derived or DEFAULT_DEPTH_TOPIC


def _choose_camera_info_topic(info_topics, rgb_topic):
    derived = _derive_related_topic(
        rgb_topic,
        "/color/image_raw",
        "/color/camera_info",
    )
    for candidate in (derived, DEFAULT_CAMERA_INFO_TOPIC):
        if candidate and candidate in info_topics:
            return candidate
    if info_topics:
        return info_topics[0]
    return derived or DEFAULT_CAMERA_INFO_TOPIC


def _derive_related_topic(topic, old_suffix, new_suffix):
    topic = str(topic or "")
    if topic.endswith(old_suffix):
        return topic[: -len(old_suffix)] + new_suffix
    return ""


def parse_sample_metadata(output):
    """Extract useful Image/CameraInfo metadata from `ros2 topic echo --no-arr`."""
    text = str(output or "")
    width = _first_int(text, r"(?m)^width:\s*(\d+)\s*$")
    height = _first_int(text, r"(?m)^height:\s*(\d+)\s*$")
    encoding_match = re.search(r"(?m)^encoding:\s*['\"]?([^'\"\n]+)", text)
    encoding = encoding_match.group(1).strip() if encoding_match else ""

    stamp_match = re.search(
        r"stamp:\s*\n\s*sec:\s*(-?\d+)\s*\n\s*nanosec:\s*(\d+)",
        text,
    )
    stamp_seconds = None
    if stamp_match:
        stamp_seconds = int(stamp_match.group(1))
        stamp_seconds += int(stamp_match.group(2)) / 1_000_000_000.0

    return {
        "width": width,
        "height": height,
        "encoding": encoding,
        "stamp_seconds": stamp_seconds,
    }


def _first_int(text, pattern):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def parse_average_rate(output):
    """Return the most recent average rate emitted by `ros2 topic hz`."""
    matches = re.findall(r"average rate:\s*([0-9]+(?:\.[0-9]+)?)", str(output or ""))
    if not matches:
        return None
    return float(matches[-1])


def format_age(age_ms):
    if age_ms is None:
        return "—"
    if age_ms < 1000:
        return f"{age_ms:.0f} ms"
    return f"{age_ms / 1000.0:.1f} s"


class _CameraHealthSignals(QObject):
    success = Signal(dict)
    error = Signal(str)
    finished = Signal()


class CameraHealthWorker(QObject):
    """Probe ROS camera streams without blocking the Qt event loop."""

    def __init__(self, selected_rgb, usecase_mode, command_timeout=3.0):
        super().__init__()
        self.selected_rgb = str(selected_rgb or "").strip()
        self.usecase_mode = int(usecase_mode)
        self.command_timeout = float(command_timeout)
        self.signals = _CameraHealthSignals()

    @Slot()
    def run(self):
        started = time.time()
        try:
            result = self._collect_health()
            result["duration_s"] = time.time() - started
            self.signals.success.emit(result)
        except Exception as exc:
            self.signals.error.emit(f"Camera health check failed: {exc}")
        finally:
            self.signals.finished.emit()

    def _collect_health(self):
        ros2_path = shutil.which("ros2")
        ros_distro = os.getenv("ROS_DISTRO", "")
        if not ros2_path:
            return {
                "ros_connected": False,
                "ros_distro": ros_distro,
                "ros_error": "ros2 command not found in PATH",
                "streams": {},
                "image_topics": [],
                "camera_info_topics": [],
                "selected_rgb": self.selected_rgb,
                "usecase_mode": self.usecase_mode,
            }

        topic_result = subprocess.run(
            [ros2_path, "topic", "list", "-t"],
            capture_output=True,
            text=True,
            check=False,
            timeout=self.command_timeout,
        )
        if topic_result.returncode != 0:
            error = topic_result.stderr.strip() or "ROS graph query failed"
            return {
                "ros_connected": False,
                "ros_distro": ros_distro,
                "ros_error": error,
                "streams": {},
                "image_topics": [],
                "camera_info_topics": [],
                "selected_rgb": self.selected_rgb,
                "usecase_mode": self.usecase_mode,
            }

        topic_types = parse_topic_list(topic_result.stdout)
        inferred = infer_camera_topics(topic_types, self.selected_rgb)
        targets = (
            ("RGB", inferred["rgb"], IMAGE_TYPE),
            ("Depth", inferred["depth"], IMAGE_TYPE),
            ("CameraInfo", inferred["camera_info"], CAMERA_INFO_TYPE),
        )

        streams = {}
        for name, topic, expected_type in targets:
            health = self._probe_stream(name, topic, expected_type, topic_types)
            streams[name.lower()] = asdict(health) | {"state": health.state}

        return {
            "ros_connected": True,
            "ros_distro": ros_distro,
            "ros_error": "",
            "streams": streams,
            "image_topics": inferred["image_topics"],
            "camera_info_topics": inferred["camera_info_topics"],
            "selected_rgb": inferred["rgb"],
            "usecase_mode": self.usecase_mode,
        }

    def _probe_stream(self, name, topic, expected_type, topic_types):
        health = StreamHealth(
            name=name,
            topic=topic,
            expected_type=expected_type,
            graph_present=bool(topic and topic in topic_types),
            message_type=topic_types.get(topic, ""),
        )
        if name == "Depth":
            health.aligned = "aligned_depth" in str(topic).lower()
        if not health.graph_present:
            health.detail = "Topic not present on the ROS 2 graph."
            return health

        sample_output, sample_error = self._echo_one(topic)
        if sample_output:
            health.sample_received = True
            metadata = parse_sample_metadata(sample_output)
            health.width = metadata["width"]
            health.height = metadata["height"]
            health.encoding = metadata["encoding"]
            health.stamp_age_ms = self._stamp_age_ms(metadata["stamp_seconds"])
        else:
            health.detail = sample_error or "No sample received before timeout."
            return health

        rate_output = self._measure_rate(topic)
        health.rate_hz = parse_average_rate(rate_output)
        return health

    def _echo_one(self, topic):
        command = [
            "ros2",
            "topic",
            "echo",
            topic,
            "--once",
            "--no-arr",
            "--qos-reliability",
            "best_effort",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.command_timeout,
            )
        except subprocess.TimeoutExpired:
            return "", "Topic exists but no message arrived before timeout."
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout, ""

        # Some ros2cli versions do not expose --no-arr. Fall back to header only.
        fallback = [
            "ros2",
            "topic",
            "echo",
            topic,
            "--once",
            "--field",
            "header",
            "--qos-reliability",
            "best_effort",
        ]
        try:
            result = subprocess.run(
                fallback,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.command_timeout,
            )
        except subprocess.TimeoutExpired:
            return "", "Topic exists but no message arrived before timeout."
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout, ""
        return "", result.stderr.strip() or "Unable to sample topic."

    def _measure_rate(self, topic):
        process = subprocess.Popen(
            ["ros2", "topic", "hz", topic, "--window", "20"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            output, _ = process.communicate(timeout=2.2)
            return output
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                output, _ = process.communicate(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate()
            return output

    @staticmethod
    def _stamp_age_ms(stamp_seconds):
        if not stamp_seconds or stamp_seconds <= 0:
            return None
        age_s = time.time() - stamp_seconds
        # A negative/huge age usually indicates simulation or a different clock domain.
        if age_s < -2.0 or age_s > 3600.0:
            return None
        return max(0.0, age_s * 1000.0)


class CameraAssistantWindow(QWidget):
    """Operator-facing RGB/depth/CameraInfo health surface."""

    health_updated = Signal(dict)

    def __init__(self, deploy_window, parent=None):
        super().__init__(parent)
        self.deploy_window = deploy_window
        self._worker = None
        self._thread = None
        self._stream_widgets = {}

        self.setObjectName("cameraAssistant")
        self.setWindowTitle("EPD Camera Assistant")
        self.resize(1040, 760)
        self.setMinimumSize(820, 620)
        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame(self)
        header.setObjectName("assistantHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 18, 24, 16)
        header_layout.setSpacing(14)

        title_stack = QVBoxLayout()
        title_stack.setSpacing(2)
        title = QLabel("Camera Assistant", header)
        title.setObjectName("assistantTitle")
        subtitle = QLabel(
            "Check ROS 2, RGB, aligned depth and CameraInfo before running perception.",
            header,
        )
        subtitle.setObjectName("assistantSubtitle")
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header_layout.addLayout(title_stack, 1)

        self.overall_badge = QLabel("NOT CHECKED", header)
        self.overall_badge.setObjectName("assistantBadge")
        self.overall_badge.setAlignment(Qt.AlignCenter)
        self.overall_badge.setMinimumSize(128, 30)
        header_layout.addWidget(self.overall_badge, 0, Qt.AlignTop)
        root.addWidget(header)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget(scroll)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(22, 18, 22, 18)
        body_layout.setSpacing(14)

        summary_card = self._card(body, "CAMERA HEALTH", "Live diagnostic snapshot")
        summary_layout = summary_card.layout()
        self.ros_value = self._summary_line(summary_layout, "ROS 2")
        self.selected_value = self._summary_line(summary_layout, "Selected RGB")
        self.mode_value = self._summary_line(summary_layout, "Perception mode")
        body_layout.addWidget(summary_card)

        streams_grid = QGridLayout()
        streams_grid.setHorizontalSpacing(12)
        streams_grid.setVerticalSpacing(12)
        for column, key in enumerate(("rgb", "depth", "camerainfo")):
            title_text = {
                "rgb": "RGB",
                "depth": "ALIGNED DEPTH",
                "camerainfo": "CAMERAINFO",
            }[key]
            card = self._stream_card(body, key, title_text)
            streams_grid.addWidget(card, 0, column)
            streams_grid.setColumnStretch(column, 1)
        body_layout.addLayout(streams_grid)

        topics_card = self._card(
            body,
            "DETECTED CAMERA TOPICS",
            "sensor_msgs/Image and CameraInfo topics found on the ROS 2 graph",
        )
        self.topics_view = QTextBrowser(topics_card)
        self.topics_view.setObjectName("topicsView")
        self.topics_view.setMinimumHeight(150)
        topics_card.layout().addWidget(self.topics_view)
        body_layout.addWidget(topics_card)

        action_card = self._card(
            body,
            "WHAT TO DO NEXT",
            "Actionable recovery guidance based on this health check",
        )
        self.remediation_label = QLabel(
            "Click Refresh health to inspect the current ROS 2 camera state.",
            action_card,
        )
        self.remediation_label.setObjectName("remediationText")
        self.remediation_label.setWordWrap(True)
        action_card.layout().addWidget(self.remediation_label)
        body_layout.addWidget(action_card)

        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        footer = QFrame(self)
        footer.setObjectName("assistantFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(22, 12, 22, 12)
        self.status_label = QLabel("No diagnostic has been run.", footer)
        self.status_label.setObjectName("assistantStatus")
        footer_layout.addWidget(self.status_label, 1)

        self.refresh_button = QPushButton("Refresh health", footer)
        self.refresh_button.setObjectName("primaryButton")
        self.refresh_button.clicked.connect(self.refresh_health)
        footer_layout.addWidget(self.refresh_button)

        close_button = QPushButton("Close", footer)
        close_button.clicked.connect(self.close)
        footer_layout.addWidget(close_button)
        root.addWidget(footer)

    def _card(self, parent, title_text, subtitle_text):
        card = QFrame(parent)
        card.setObjectName("assistantCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(9)
        title = QLabel(title_text, card)
        title.setObjectName("cardTitle")
        subtitle = QLabel(subtitle_text, card)
        subtitle.setObjectName("cardSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return card

    def _summary_line(self, layout, name):
        row = QHBoxLayout()
        label = QLabel(name)
        label.setObjectName("fieldName")
        label.setMinimumWidth(112)
        value = QLabel("—")
        value.setObjectName("fieldValue")
        value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(label)
        row.addWidget(value, 1)
        layout.addLayout(row)
        return value

    def _stream_card(self, parent, key, title_text):
        card = self._card(parent, title_text, "Waiting for health check")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = card.layout()

        state = QLabel("Not checked", card)
        state.setObjectName("streamState")
        state.setAlignment(Qt.AlignCenter)
        state.setMinimumHeight(28)
        layout.addWidget(state)

        topic = QLabel("—", card)
        topic.setObjectName("streamTopic")
        topic.setWordWrap(True)
        topic.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(topic)

        details = QLabel("Resolution: —\nRate: —\nLast frame: —", card)
        details.setObjectName("streamDetails")
        details.setWordWrap(True)
        layout.addWidget(details)

        self._stream_widgets[key] = {
            "card": card,
            "state": state,
            "topic": topic,
            "details": details,
        }
        return card

    @Slot()
    def refresh_health(self):
        if self._thread is not None and self._thread.isRunning():
            return

        selected_rgb = self.deploy_window.topic_button.currentText().strip()
        selected_rgb = selected_rgb or str(
            getattr(self.deploy_window, "_input_image_topic", "") or ""
        ).strip()
        usecase_mode = int(getattr(self.deploy_window, "usecase_mode", 0))

        self.refresh_button.setEnabled(False)
        self.overall_badge.setText("CHECKING")
        self._set_state_property(self.overall_badge, "checking")
        self.status_label.setText("Checking ROS 2 graph and sampling camera streams…")

        self._worker = CameraHealthWorker(selected_rgb, usecase_mode)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.signals.success.connect(self._on_health_success)
        self._worker.signals.error.connect(self._on_health_error)
        self._worker.signals.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_worker)
        self._thread.start()

    @Slot(dict)
    def _on_health_success(self, result):
        self._render_result(result)
        self.health_updated.emit(result)

    @Slot(str)
    def _on_health_error(self, message):
        self.overall_badge.setText("ERROR")
        self._set_state_property(self.overall_badge, "error")
        self.status_label.setText(message)
        self.remediation_label.setText(
            "Retry the check. If it fails again, verify ROS 2 is sourced and run "
            "`ros2 topic list -t` in a terminal."
        )

    @Slot()
    def _clear_worker(self):
        self.refresh_button.setEnabled(True)
        self._worker = None
        self._thread = None

    def _render_result(self, result):
        connected = bool(result.get("ros_connected"))
        ros_distro = result.get("ros_distro") or "unknown distro"
        if connected:
            self.ros_value.setText(f"Connected ({ros_distro})")
        else:
            error = result.get("ros_error") or "ROS graph unavailable"
            self.ros_value.setText(f"Unavailable — {error}")

        selected_rgb = result.get("selected_rgb") or "Not configured"
        self.selected_value.setText(selected_rgb)
        mode = int(result.get("usecase_mode", 0))
        self.mode_value.setText(self._mode_name(mode))

        streams = result.get("streams", {})
        self._render_stream("rgb", streams.get("rgb", {}), required=True)
        requires_3d = mode in (3, 4)
        self._render_stream("depth", streams.get("depth", {}), required=requires_3d)
        self._render_stream(
            "camerainfo",
            streams.get("camerainfo", {}),
            required=requires_3d,
        )

        self._render_topics(result)
        overall_state = self._overall_state(result, requires_3d)
        self._render_overall(overall_state)
        self.remediation_label.setText(self._remediation(result, requires_3d))
        duration = float(result.get("duration_s", 0.0) or 0.0)
        self.status_label.setText(f"Health check completed in {duration:.1f} s.")

    def _render_stream(self, key, stream, required):
        widgets = self._stream_widgets[key]
        state = stream.get("state", "missing")
        state_text = {
            "live": "✓ Live",
            "unresponsive": "⚠ No sample",
            "missing": "✕ Missing",
        }.get(state, state.title())
        if not required:
            state_text += "  •  optional"
        widgets["state"].setText(state_text)
        self._set_state_property(widgets["state"], state)

        topic = stream.get("topic") or "Not found"
        widgets["topic"].setText(topic)
        widgets["topic"].setToolTip(topic)

        width = stream.get("width")
        height = stream.get("height")
        resolution = f"{width}×{height}" if width and height else "—"
        rate = stream.get("rate_hz")
        rate_text = f"{rate:.1f} Hz" if rate is not None else "—"
        age_text = format_age(stream.get("stamp_age_ms"))
        encoding = stream.get("encoding") or "—"

        detail_lines = [
            f"Resolution: {resolution}",
            f"Rate: {rate_text}",
            f"Last frame age: {age_text}",
            f"Encoding: {encoding}",
        ]
        if key == "depth":
            aligned = stream.get("aligned")
            detail_lines.append(
                "Alignment: aligned to colour" if aligned else "Alignment: not verified"
            )
        detail = stream.get("detail")
        if detail:
            detail_lines.append(detail)
        widgets["details"].setText("\n".join(detail_lines))

    def _render_topics(self, result):
        image_topics = result.get("image_topics", [])
        info_topics = result.get("camera_info_topics", [])
        lines = ["Image topics:"]
        lines.extend(f"  {topic}" for topic in image_topics)
        if not image_topics:
            lines.append("  (none detected)")
        lines.append("")
        lines.append("CameraInfo topics:")
        lines.extend(f"  {topic}" for topic in info_topics)
        if not info_topics:
            lines.append("  (none detected)")
        self.topics_view.setPlainText("\n".join(lines))

    def _render_overall(self, state):
        text = {
            "ready": "CAMERA READY",
            "partial": "PARTIAL",
            "missing": "NOT READY",
            "ros_error": "ROS UNAVAILABLE",
        }[state]
        self.overall_badge.setText(text)
        self._set_state_property(self.overall_badge, state)

    def _overall_state(self, result, requires_3d):
        if not result.get("ros_connected"):
            return "ros_error"
        streams = result.get("streams", {})
        rgb_live = streams.get("rgb", {}).get("state") == "live"
        if not rgb_live:
            return "missing"
        if requires_3d:
            depth_live = streams.get("depth", {}).get("state") == "live"
            info_live = streams.get("camerainfo", {}).get("state") == "live"
            if depth_live and info_live:
                return "ready"
            return "partial"
        return "ready"

    def _remediation(self, result, requires_3d):
        if not result.get("ros_connected"):
            return (
                "ROS 2 is not available to the GUI. Source /opt/ros/<distro>/setup.bash and "
                "your workspace, then relaunch EPD. Confirm `ros2 topic list -t` works."
            )

        streams = result.get("streams", {})
        rgb = streams.get("rgb", {})
        depth = streams.get("depth", {})
        info = streams.get("camerainfo", {})
        if rgb.get("state") != "live":
            return (
                "RGB is required. Start the camera node, confirm the selected topic publishes "
                "sensor_msgs/msg/Image, then return to Deploy and Refresh topics. For a "
                "RealSense D435i, the normal RGB topic is " + DEFAULT_RGB_TOPIC + "."
            )
        if requires_3d and depth.get("state") != "live":
            return (
                "Localization/Tracking requires live depth. Enable aligned depth on the camera "
                "driver and confirm " + DEFAULT_DEPTH_TOPIC + " (or your equivalent) is live."
            )
        if requires_3d and info.get("state") != "live":
            return (
                "Localization/Tracking requires CameraInfo for camera intrinsics. Confirm the "
                "camera driver publishes " + DEFAULT_CAMERA_INFO_TOPIC + " or an equivalent."
            )
        if requires_3d and not depth.get("aligned", False):
            return (
                "Depth is live, but alignment to the colour frame was not verified from the "
                "topic name. Use an aligned-depth stream for reliable RGB/depth geometry."
            )
        return (
            "Camera inputs look healthy for the selected mode. Return to Deploy, validate the "
            "model/labels, and run perception."
        )

    @staticmethod
    def _mode_name(mode):
        names = {
            0: "Classification",
            1: "Counting",
            2: "Color-Matching",
            3: "Localization (3D)",
            4: "Tracking (3D + persistent IDs)",
        }
        return names.get(mode, f"Mode {mode}")

    @staticmethod
    def _set_state_property(widget, state):
        widget.setProperty("healthState", state)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QWidget#cameraAssistant {
                background-color: #10151d;
                color: #edf2f8;
            }
            QFrame#assistantHeader,
            QFrame#assistantFooter {
                background-color: #111720;
                border: 0;
            }
            QLabel#assistantTitle {
                font-size: 21px;
                font-weight: 700;
                color: #f4f7fb;
            }
            QLabel#assistantSubtitle,
            QLabel#cardSubtitle,
            QLabel#assistantStatus {
                color: #96a4b7;
            }
            QFrame#assistantCard {
                background-color: #171e28;
                border: 1px solid #2b3645;
                border-radius: 12px;
            }
            QLabel#cardTitle {
                color: #91a0b6;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#fieldName {
                color: #91a0b6;
            }
            QLabel#fieldValue,
            QLabel#streamTopic {
                color: #e8edf5;
            }
            QLabel#streamDetails,
            QLabel#remediationText {
                color: #b7c1cf;
                line-height: 1.25;
            }
            QTextBrowser#topicsView {
                background-color: #111720;
                color: #bdc8d7;
                border: 1px solid #2a3544;
                border-radius: 8px;
            }
            QLabel#assistantBadge,
            QLabel#streamState {
                border: 1px solid #465267;
                border-radius: 7px;
                padding: 4px 9px;
                font-weight: 700;
            }
            QLabel[healthState="ready"],
            QLabel[healthState="live"] {
                color: #a7e6b0;
                background-color: #15351e;
                border-color: #2d7140;
            }
            QLabel[healthState="partial"],
            QLabel[healthState="unresponsive"],
            QLabel[healthState="checking"] {
                color: #eadb9f;
                background-color: #342e18;
                border-color: #756229;
            }
            QLabel[healthState="missing"],
            QLabel[healthState="error"],
            QLabel[healthState="ros_error"] {
                color: #efaaaa;
                background-color: #381b1e;
                border-color: #77333b;
            }
            QPushButton {
                min-height: 34px;
                padding: 5px 13px;
                border: 1px solid #364357;
                border-radius: 8px;
                background-color: #1b2430;
                color: #dbe3ee;
            }
            QPushButton:hover {
                background-color: #253143;
                border-color: #60738f;
            }
            QPushButton#primaryButton {
                background-color: #3854c8;
                border-color: #526ee0;
                color: #ffffff;
                font-weight: 700;
            }
            """
        )
