"""EPD-2 productization: embedded live perception preview and telemetry."""

from __future__ import annotations

import os
import threading
import time
from collections import deque

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from windows.job_controller import JobState

try:
    import rclpy
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage, Image
    _ROS_IMAGE_AVAILABLE = True
except ImportError:
    _ROS_IMAGE_AVAILABLE = False

try:
    from epd_msgs.msg import (
        EPDObjectDetection,
        EPDObjectLocalization,
        EPDObjectTracking,
    )
    _EPD_MSGS_AVAILABLE = True
except ImportError:
    _EPD_MSGS_AVAILABLE = False


OUTPUT_IMAGE_TOPIC = "/easy_perception_deployment/image_output"
_FRAME_STALE_SEC = 2.5
_PREVIEW_STORE_INTERVAL_SEC = 0.06


def transport_topic(base_topic, transport):
    """Return the ROS image topic used for the selected image transport."""
    base = str(base_topic or "").strip().rstrip("/")
    if not base:
        return ""
    if str(transport or "raw").lower() == "compressed":
        if base.endswith("/compressed"):
            return base
        return base + "/compressed"
    return base


def select_preview_source(input_topic, transport, perception_active, overlay_enabled):
    """Select the operator preview source without changing Deploy configuration."""
    if perception_active and overlay_enabled:
        return {
            "topic": transport_topic(OUTPUT_IMAGE_TOPIC, transport),
            "source": "Detection overlay",
            "compressed": str(transport).lower() == "compressed",
        }
    return {
        "topic": transport_topic(input_topic, transport),
        "source": "Camera RGB",
        "compressed": str(transport).lower() == "compressed",
    }


def object_count_from_message(message):
    """Return a mode-independent object count from an EPD output message."""
    if hasattr(message, "objects"):
        try:
            return len(message.objects)
        except TypeError:
            return None
    if hasattr(message, "class_indices"):
        try:
            return len(message.class_indices)
        except TypeError:
            return None
    return None


def comparable_header_age_ms(message, now=None, max_age_sec=60.0):
    """Return wall-clock message age only when the ROS stamp is comparable."""
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = int(getattr(stamp, "sec", 0) or 0)
    nanosec = int(getattr(stamp, "nanosec", 0) or 0)
    if sec == 0 and nanosec == 0:
        return None
    stamp_sec = sec + nanosec / 1e9
    wall_now = time.time() if now is None else float(now)
    age_sec = wall_now - stamp_sec
    if age_sec < 0 or age_sec > max_age_sec:
        return None
    return age_sec * 1000.0


def raw_packet_from_message(message):
    """Copy a sensor_msgs/Image into a Qt-independent packet."""
    width = int(getattr(message, "width", 0) or 0)
    height = int(getattr(message, "height", 0) or 0)
    step = int(getattr(message, "step", 0) or 0)
    encoding = str(getattr(message, "encoding", "") or "").lower()
    data = bytes(getattr(message, "data", b"") or b"")
    if width <= 0 or height <= 0 or step <= 0 or not data:
        return None
    if len(data) < step * height:
        return None
    return {
        "kind": "raw",
        "width": width,
        "height": height,
        "step": step,
        "encoding": encoding,
        "data": data,
    }


def compressed_packet_from_message(message):
    """Copy a sensor_msgs/CompressedImage into a Qt-independent packet."""
    data = bytes(getattr(message, "data", b"") or b"")
    if not data:
        return None
    return {
        "kind": "compressed",
        "format": str(getattr(message, "format", "") or ""),
        "data": data,
    }


def qimage_from_packet(packet):
    """Convert a preview packet to a detached QImage on the GUI thread."""
    if not packet:
        return None, "No frame data"
    if packet.get("kind") == "compressed":
        image = QImage.fromData(packet.get("data", b""))
        if image.isNull():
            return None, "Unable to decode compressed preview image"
        return image.copy(), ""

    width = int(packet.get("width", 0) or 0)
    height = int(packet.get("height", 0) or 0)
    step = int(packet.get("step", 0) or 0)
    data = packet.get("data", b"")
    encoding = str(packet.get("encoding", "") or "").lower()

    formats = {
        "rgb8": (QImage.Format_RGB888, False),
        "bgr8": (QImage.Format_BGR888, False),
        "rgba8": (QImage.Format_RGBA8888, False),
        "bgra8": (QImage.Format_RGBA8888, True),
        "mono8": (QImage.Format_Grayscale8, False),
    }
    image_format = formats.get(encoding)
    if image_format is None:
        return None, f"Unsupported preview encoding: {encoding or 'unknown'}"

    fmt, swap_rgb = image_format
    image = QImage(data, width, height, step, fmt).copy()
    if image.isNull():
        return None, "Unable to construct preview image"
    if swap_rgb:
        image = image.rgbSwapped()
    return image, ""


def _rate_from_stamps(stamps):
    if len(stamps) < 2:
        return None
    delta = stamps[-1] - stamps[0]
    if delta <= 0:
        return None
    return (len(stamps) - 1) / delta


class LivePerceptionMonitor:
    """Background ROS subscriber for image preview and mode telemetry."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._config_revision = 0
        self._config = {
            "input_topic": "",
            "transport": "raw",
            "perception_active": False,
            "overlay_enabled": True,
            "usecase_mode": 0,
        }
        self._latest = self._empty_snapshot()
        self._frame_stamps = deque(maxlen=30)
        self._telemetry_stamps = deque(maxlen=30)
        self._last_frame_store = 0.0

    @staticmethod
    def _empty_snapshot():
        return {
            "frame_seq": 0,
            "frame": None,
            "frame_monotonic": 0.0,
            "frame_fps": None,
            "telemetry_fps": None,
            "latency_ms": None,
            "object_count": None,
            "telemetry_monotonic": 0.0,
            "source": "Camera RGB",
            "source_topic": "",
            "error": "",
            "telemetry_available": _EPD_MSGS_AVAILABLE,
        }

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="EPDLivePerceptionPreview",
        )
        self._thread.start()

    def stop(self, timeout_sec=1.5):
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_sec)
        self._thread = None

    def configure(
            self,
            input_topic,
            transport,
            perception_active,
            overlay_enabled,
            usecase_mode):
        new_config = {
            "input_topic": str(input_topic or "").strip(),
            "transport": str(transport or "raw").lower(),
            "perception_active": bool(perception_active),
            "overlay_enabled": bool(overlay_enabled),
            "usecase_mode": int(usecase_mode),
        }
        with self._lock:
            if new_config == self._config:
                return
            self._config = new_config
            self._config_revision += 1

    def snapshot(self):
        with self._lock:
            return dict(self._latest)

    def _run(self):
        if not _ROS_IMAGE_AVAILABLE:
            self._set_error(
                "ROS Python image support is unavailable. Source ROS 2 and the "
                "EPD workspace before launching the GUI."
            )
            return

        context = None
        node = None
        subscriptions = []
        applied_revision = -1
        try:
            context = rclpy.context.Context()
            context.init(args=None)
            node = rclpy.create_node("epd_live_perception_preview", context=context)
            while not self._stop_event.is_set() and context.ok():
                with self._lock:
                    revision = self._config_revision
                    config = dict(self._config)
                if revision != applied_revision:
                    for subscription in subscriptions:
                        node.destroy_subscription(subscription)
                    subscriptions = self._create_subscriptions(node, config)
                    applied_revision = revision
                rclpy.spin_once(node, timeout_sec=0.08)
        except Exception as exc:
            self._set_error(f"Live preview ROS subscriber failed: {exc}")
        finally:
            if node is not None:
                for subscription in subscriptions:
                    try:
                        node.destroy_subscription(subscription)
                    except Exception:
                        pass
                try:
                    node.destroy_node()
                except Exception:
                    pass
            if context is not None:
                try:
                    context.shutdown()
                except Exception:
                    pass

    def _create_subscriptions(self, node, config):
        source = select_preview_source(
            config["input_topic"],
            config["transport"],
            config["perception_active"],
            config["overlay_enabled"],
        )
        with self._lock:
            old_seq = self._latest.get("frame_seq", 0)
            self._latest = self._empty_snapshot()
            self._latest["frame_seq"] = old_seq
            self._latest["source"] = source["source"]
            self._latest["source_topic"] = source["topic"]
        self._frame_stamps.clear()
        self._telemetry_stamps.clear()
        self._last_frame_store = 0.0

        subscriptions = []
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        if source["topic"]:
            msg_type = CompressedImage if source["compressed"] else Image
            callback = (
                self._compressed_image_callback
                if source["compressed"]
                else self._raw_image_callback
            )
            subscriptions.append(
                node.create_subscription(msg_type, source["topic"], callback, qos)
            )

        if config["perception_active"] and _EPD_MSGS_AVAILABLE:
            telemetry = self._telemetry_spec(config["usecase_mode"])
            if telemetry is not None:
                topic, msg_type = telemetry
                subscriptions.append(
                    node.create_subscription(
                        msg_type,
                        topic,
                        self._telemetry_callback,
                        qos,
                    )
                )
        return subscriptions

    @staticmethod
    def _telemetry_spec(usecase_mode):
        if not _EPD_MSGS_AVAILABLE:
            return None
        if usecase_mode in (0, 1):
            return (
                "/easy_perception_deployment/epd_p2_output",
                EPDObjectDetection,
            )
        if usecase_mode == 2:
            return (
                "/easy_perception_deployment/epd_p3_output",
                EPDObjectDetection,
            )
        if usecase_mode == 3:
            return (
                "/easy_perception_deployment/epd_localize_output",
                EPDObjectLocalization,
            )
        if usecase_mode == 4:
            return (
                "/easy_perception_deployment/epd_tracking_output",
                EPDObjectTracking,
            )
        return None

    def _raw_image_callback(self, message):
        packet = raw_packet_from_message(message)
        self._store_frame(packet, message)

    def _compressed_image_callback(self, message):
        packet = compressed_packet_from_message(message)
        self._store_frame(packet, message)

    def _store_frame(self, packet, message):
        if packet is None:
            return
        monotonic_now = time.monotonic()
        self._frame_stamps.append(monotonic_now)
        if monotonic_now - self._last_frame_store < _PREVIEW_STORE_INTERVAL_SEC:
            return
        self._last_frame_store = monotonic_now
        frame_fps = _rate_from_stamps(self._frame_stamps)
        frame_age = comparable_header_age_ms(message)
        with self._lock:
            self._latest["frame_seq"] += 1
            self._latest["frame"] = packet
            self._latest["frame_monotonic"] = monotonic_now
            self._latest["frame_fps"] = frame_fps
            if self._latest.get("latency_ms") is None and frame_age is not None:
                self._latest["latency_ms"] = frame_age
            self._latest["error"] = ""

    def _telemetry_callback(self, message):
        monotonic_now = time.monotonic()
        self._telemetry_stamps.append(monotonic_now)
        fps = _rate_from_stamps(self._telemetry_stamps)
        count = object_count_from_message(message)
        process_time = getattr(message, "process_time", None)
        latency = None
        if process_time is not None:
            try:
                latency = float(process_time)
            except (TypeError, ValueError):
                latency = None
        if latency is None or latency <= 0:
            latency = comparable_header_age_ms(message)
        with self._lock:
            self._latest["telemetry_fps"] = fps
            self._latest["object_count"] = count
            self._latest["latency_ms"] = latency
            self._latest["telemetry_monotonic"] = monotonic_now

    def _set_error(self, message):
        with self._lock:
            self._latest["error"] = str(message)


class _PreviewSurface(QLabel):
    """Image surface that preserves aspect ratio as Deploy is resized."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = None
        self.setObjectName("livePreviewSurface")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(520, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setText("Camera preview will appear here.")

    def set_image(self, image):
        self._image = image.copy() if image is not None else None
        self._render()

    def set_placeholder(self, text):
        self._image = None
        self.clear()
        self.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def _render(self):
        if self._image is None or self._image.isNull():
            return
        pixmap = QPixmap.fromImage(self._image)
        scaled = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)


class _LivePerceptionController(QObject):
    """Connect the ROS preview monitor to the refreshed Deploy UI."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.deploy = main_window.deploy_window
        self.monitor = LivePerceptionMonitor()
        self._last_frame_seq = -1
        self._monitor_started = False
        self._disabled = os.getenv("EPD_DISABLE_LIVE_PREVIEW") == "1"

        self.timer = QTimer(self)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self._tick)

        self._build_card()
        self._connect_hooks()
        self.deploy.installEventFilter(self)
        if self.deploy.isVisible():
            self._start_monitor()

    def _build_card(self):
        content = self.deploy.findChild(QWidget, "deployContent")
        if content is None or not isinstance(content.layout(), QGridLayout):
            return
        grid = content.layout()

        card = QFrame(content)
        card.setObjectName("sectionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(10)
        title_stack = QVBoxLayout()
        title_stack.setSpacing(2)
        title = QLabel("LIVE PERCEPTION", card)
        title.setObjectName("sectionTitle")
        subtitle = QLabel(
            "Camera and perception output inside Deploy. rqt_image_view is optional.",
            card,
        )
        subtitle.setObjectName("sectionSubtitle")
        subtitle.setWordWrap(True)
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header.addLayout(title_stack, 1)

        self.state_badge = QLabel("STOPPED", card)
        self.state_badge.setObjectName("previewStateBadge")
        self.state_badge.setAlignment(Qt.AlignCenter)
        self.state_badge.setMinimumWidth(88)
        header.addWidget(self.state_badge, 0, Qt.AlignTop)
        layout.addLayout(header)

        self.surface = _PreviewSurface(card)
        layout.addWidget(self.surface, 1)

        truth_row = QHBoxLayout()
        truth_row.setSpacing(8)
        self.source_label = QLabel("Source: Camera RGB", card)
        self.source_label.setObjectName("previewTruth")
        self.overlay_label = QLabel("Detection overlay: —", card)
        self.overlay_label.setObjectName("previewTruth")
        self.mask_label = QLabel("Object masks: —", card)
        self.mask_label.setObjectName("previewTruth")
        truth_row.addWidget(self.source_label)
        truth_row.addWidget(self.overlay_label)
        truth_row.addWidget(self.mask_label)
        truth_row.addStretch(1)
        layout.addLayout(truth_row)

        metrics = QHBoxLayout()
        metrics.setSpacing(8)
        self.object_value = self._metric_chip(card, metrics, "OBJECTS")
        self.fps_value = self._metric_chip(card, metrics, "FPS")
        self.latency_value = self._metric_chip(card, metrics, "LATENCY")
        self.frame_age_value = self._metric_chip(card, metrics, "FRAME AGE")
        layout.addLayout(metrics)

        self.hint_label = QLabel(
            "Perception is stopped. Camera preview stays available while Deploy is open.",
            card,
        )
        self.hint_label.setObjectName("previewHint")
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        row = grid.rowCount()
        grid.addWidget(card, row, 0, 1, 2)
        self.deploy._epd2_preview_card = card
        self._append_style()

    @staticmethod
    def _metric_chip(parent, layout, title):
        frame = QFrame(parent)
        frame.setObjectName("previewMetric")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(10, 7, 10, 7)
        frame_layout.setSpacing(1)
        name = QLabel(title, frame)
        name.setObjectName("previewMetricName")
        value = QLabel("—", frame)
        value.setObjectName("previewMetricValue")
        frame_layout.addWidget(name)
        frame_layout.addWidget(value)
        layout.addWidget(frame, 1)
        return value

    def _connect_hooks(self):
        self.deploy._job_controller.state_changed.connect(self._on_job_state)
        self.deploy.topic_button.currentTextChanged.connect(self._schedule_configure)
        self.deploy.transport_combo.currentTextChanged.connect(self._schedule_configure)
        self.deploy.visualize_button.clicked.connect(self._schedule_configure)
        self.deploy.segmentation_button.clicked.connect(self._schedule_configure)
        self.deploy.usecase_config_button.currentTextChanged.connect(
            self._schedule_configure
        )

    def _schedule_configure(self, *args):
        QTimer.singleShot(0, self._configure_monitor)
        QTimer.singleShot(250, self._configure_monitor)

    def _on_job_state(self, state, message):
        self._configure_monitor()
        self._tick()

    def _start_monitor(self):
        if self._disabled:
            self._set_state("UNAVAILABLE", "unavailable")
            self.surface.set_placeholder("Live preview disabled by environment.")
            self.hint_label.setText("Unset EPD_DISABLE_LIVE_PREVIEW to enable it.")
            return
        if self._monitor_started:
            return
        self.monitor.start()
        self._monitor_started = True
        self._configure_monitor()
        self.timer.start()
        self._tick()

    def _stop_monitor(self):
        self.timer.stop()
        if self._monitor_started:
            self.monitor.stop()
        self._monitor_started = False
        self._last_frame_seq = -1

    def _configure_monitor(self):
        if not self._monitor_started:
            return
        state = self.deploy._job_controller.state
        perception_active = state in (
            JobState.STARTING,
            JobState.RUNNING,
            JobState.STOPPING,
        )
        self.monitor.configure(
            input_topic=self.deploy.topic_button.currentText().strip(),
            transport=getattr(self.deploy, "_image_transport", "raw"),
            perception_active=perception_active,
            overlay_enabled=getattr(self.deploy, "visualizeFlag", False),
            usecase_mode=getattr(self.deploy, "usecase_mode", 0),
        )

    def _tick(self):
        if self._disabled or not self._monitor_started:
            return
        snapshot = self.monitor.snapshot()
        now = time.monotonic()
        frame_time = float(snapshot.get("frame_monotonic", 0.0) or 0.0)
        frame_age = now - frame_time if frame_time > 0 else None
        fresh_frame = frame_age is not None and frame_age <= _FRAME_STALE_SEC

        frame_seq = int(snapshot.get("frame_seq", 0) or 0)
        if fresh_frame and frame_seq != self._last_frame_seq:
            image, error = qimage_from_packet(snapshot.get("frame"))
            if image is not None:
                self.surface.set_image(image)
                self._last_frame_seq = frame_seq
            elif error:
                self.surface.set_placeholder(error)

        state = self.deploy._job_controller.state
        error = str(snapshot.get("error", "") or "")
        if error:
            self._set_state("UNAVAILABLE", "unavailable")
            self.surface.set_placeholder("Live preview unavailable.")
            self.hint_label.setText(error)
        elif state == JobState.STARTING:
            self._set_state("STARTING", "starting")
            self.hint_label.setText(
                "Starting perception. Waiting for the selected preview stream."
            )
        elif state == JobState.RUNNING and fresh_frame:
            self._set_state("LIVE", "live")
            self.hint_label.setText(
                "Perception is running. The view follows the configured overlay setting."
            )
        elif state == JobState.RUNNING:
            self._set_state("WAITING", "waiting")
            self.surface.set_placeholder(
                "Perception is running, but no fresh preview frame has arrived."
            )
            self.hint_label.setText(
                "Check /easy_perception_deployment/image_output, Detection overlay, "
                "the selected image transport, and camera health."
            )
        elif state == JobState.STOPPING:
            self._set_state("STOPPING", "starting")
            self.hint_label.setText("Stopping perception safely.")
        elif state == JobState.FAILED:
            self._set_state("FAILED", "unavailable")
            self.hint_label.setText(
                "Deployment failed. Review the runtime message and deploy log."
            )
        else:
            self._set_state("STOPPED", "stopped")
            if fresh_frame:
                self.hint_label.setText(
                    "Perception is stopped. Camera preview is live and ready for setup."
                )
            else:
                self.surface.set_placeholder(
                    "Perception is stopped. Waiting for the configured camera stream."
                )
                self.hint_label.setText(
                    "Start the camera or use Camera Assistant to verify the RGB stream."
                )

        self._render_truth(snapshot)
        self._render_metrics(snapshot, frame_age, state)

    def _render_truth(self, snapshot):
        source = str(snapshot.get("source", "Camera RGB"))
        topic = str(snapshot.get("source_topic", "") or "")
        self.source_label.setText(f"Source: {source}")
        self.source_label.setToolTip(topic)
        overlay = "On" if getattr(self.deploy, "visualizeFlag", False) else "Off"
        masks = (
            "On"
            if getattr(self.deploy, "publish_detection_segmentation", False)
            else "Off"
        )
        self.overlay_label.setText(f"Detection overlay: {overlay}")
        self.mask_label.setText(f"Object masks: {masks}")

    def _render_metrics(self, snapshot, frame_age, state):
        active = state in (JobState.STARTING, JobState.RUNNING, JobState.STOPPING)
        telemetry_time = float(snapshot.get("telemetry_monotonic", 0.0) or 0.0)
        telemetry_fresh = (
            telemetry_time > 0
            and time.monotonic() - telemetry_time <= _FRAME_STALE_SEC
        )
        count = snapshot.get("object_count") if active and telemetry_fresh else None
        fps = snapshot.get("telemetry_fps") if active else snapshot.get("frame_fps")
        if fps is None:
            fps = snapshot.get("frame_fps") if active else None
        latency = snapshot.get("latency_ms") if active and telemetry_fresh else None

        self.object_value.setText("—" if count is None else str(count))
        self.fps_value.setText("—" if fps is None else f"{float(fps):.1f}")
        self.latency_value.setText(
            "—" if latency is None else f"{float(latency):.1f} ms"
        )
        self.frame_age_value.setText(
            "—" if frame_age is None else f"{max(frame_age, 0.0):.1f} s"
        )

        footer_fps = "--" if fps is None else f"{float(fps):.1f}"
        footer_latency = "--" if latency is None else f"{float(latency):.1f} ms"
        footer_objects = "--" if count is None else str(count)
        self.deploy.fps_label.setText(
            f"FPS: {footer_fps} | Latency: {footer_latency} | "
            f"Objects: {footer_objects}"
        )

    def _set_state(self, text, state):
        self.state_badge.setText(text)
        self.state_badge.setProperty("previewState", state)
        self.state_badge.style().unpolish(self.state_badge)
        self.state_badge.style().polish(self.state_badge)

    def _append_style(self):
        self.deploy.setStyleSheet(
            self.deploy.styleSheet()
            + """
            QLabel#livePreviewSurface {
                color: #8d99ab;
                background-color: #090c11;
                border: 1px solid #293341;
                border-radius: 10px;
                padding: 6px;
                font-size: 11px;
            }
            QLabel#previewStateBadge {
                color: #aeb8c7;
                background-color: #141a23;
                border: 1px solid #313c4b;
                border-radius: 7px;
                padding: 5px 9px;
                font-size: 9px;
                font-weight: 700;
            }
            QLabel#previewStateBadge[previewState="live"] {
                color: #9de2b2;
                background-color: #14241b;
                border-color: #285b39;
            }
            QLabel#previewStateBadge[previewState="starting"],
            QLabel#previewStateBadge[previewState="waiting"] {
                color: #e8c98f;
                background-color: #292116;
                border-color: #634a24;
            }
            QLabel#previewStateBadge[previewState="unavailable"] {
                color: #e3a1a1;
                background-color: #2a1719;
                border-color: #653238;
            }
            QLabel#previewTruth {
                color: #8e9bad;
                background-color: #121720;
                border: 1px solid #2b3543;
                border-radius: 6px;
                padding: 4px 7px;
                font-size: 9px;
            }
            QFrame#previewMetric {
                background-color: #121720;
                border: 1px solid #2b3543;
                border-radius: 8px;
            }
            QLabel#previewMetricName {
                color: #748198;
                font-size: 8px;
                font-weight: 700;
            }
            QLabel#previewMetricValue {
                color: #e4eaf2;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#previewHint {
                color: #8491a4;
                font-size: 10px;
            }
            """
        )

    def eventFilter(self, obj, event):
        if obj is self.deploy and event.type() == QEvent.Show:
            QTimer.singleShot(0, self._start_monitor)
        elif obj is self.deploy and event.type() in (QEvent.Hide, QEvent.Close):
            self._stop_monitor()
        return super().eventFilter(obj, event)


def apply_epd2_productization(main_window):
    """Attach the EPD-2 Live Perception View to the refreshed Deploy UI."""
    if getattr(main_window, "_epd2_productization_applied", False):
        return None
    main_window._epd2_productization_applied = True
    controller = _LivePerceptionController(main_window)
    main_window._epd2_live_perception_controller = controller
    return controller
