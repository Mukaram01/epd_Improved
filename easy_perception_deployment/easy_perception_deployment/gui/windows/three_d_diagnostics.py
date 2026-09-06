"""EPD-6: manipulation-facing 3D perception diagnostics.

This module is diagnostic-only. It does not change inference, filtering, tracking,
scene ownership, planning, or robot motion.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass

from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


GEOMETRY_COUNTER_KEYS = (
    "detections_total",
    "geometry_valid_total",
    "geometry_degraded_total",
    "geometry_invalid_total",
    "invalid_intrinsics_total",
    "empty_mask_total",
    "insufficient_depth_total",
    "empty_cloud_total",
    "nonfinite_geometry_total",
)


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _stamp_ns(stamp):
    if stamp is None:
        return None
    try:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return None


def _point_tuple(point):
    return (
        float(getattr(point, "x", float("nan"))),
        float(getattr(point, "y", float("nan"))),
        float(getattr(point, "z", float("nan"))),
    )


def _vector_norm(vector):
    values = _point_tuple(vector)
    if not all(_finite(value) for value in values):
        return float("nan")
    return math.sqrt(sum(value * value for value in values))


def _quaternion_norm(quaternion):
    values = (
        getattr(quaternion, "x", float("nan")),
        getattr(quaternion, "y", float("nan")),
        getattr(quaternion, "z", float("nan")),
        getattr(quaternion, "w", float("nan")),
    )
    if not all(_finite(value) for value in values):
        return float("nan")
    return math.sqrt(sum(float(value) ** 2 for value in values))


def inspect_frame_contract(message):
    """Return truthful alignment/intrinsics checks for a P3 result message."""
    width = int(getattr(message, "frame_width", 0) or 0)
    height = int(getattr(message, "frame_height", 0) or 0)
    depth = getattr(message, "depth_image", None)
    depth_width = int(getattr(depth, "width", 0) or 0)
    depth_height = int(getattr(depth, "height", 0) or 0)
    encoding = str(getattr(depth, "encoding", "") or "")

    intrinsics = {
        "fx": getattr(message, "fx", None),
        "fy": getattr(message, "fy", None),
        "ppx": getattr(message, "ppx", None),
        "ppy": getattr(message, "ppy", None),
    }
    intrinsics_ok = (
        all(_finite(value) for value in intrinsics.values())
        and float(intrinsics["fx"]) > 0.0
        and float(intrinsics["fy"]) > 0.0
    )
    shape_ok = (
        width > 0
        and height > 0
        and depth_width == width
        and depth_height == height
    )
    encoding_ok = encoding in ("16UC1", "32FC1")

    result_stamp = _stamp_ns(getattr(getattr(message, "header", None), "stamp", None))
    depth_stamp = _stamp_ns(getattr(getattr(depth, "header", None), "stamp", None))
    stamps_comparable = result_stamp is not None and depth_stamp is not None
    stamps_match = stamps_comparable and result_stamp == depth_stamp

    if intrinsics_ok and shape_ok and encoding_ok and stamps_match:
        state = "aligned"
    elif not intrinsics_ok or not shape_ok or not encoding_ok:
        state = "invalid"
    else:
        state = "warning"

    return {
        "state": state,
        "intrinsics_ok": intrinsics_ok,
        "shape_ok": shape_ok,
        "encoding_ok": encoding_ok,
        "stamps_match": stamps_match,
        "stamps_comparable": stamps_comparable,
        "frame_width": width,
        "frame_height": height,
        "depth_width": depth_width,
        "depth_height": depth_height,
        "depth_encoding": encoding,
        "result_stamp_ns": result_stamp,
        "depth_stamp_ns": depth_stamp,
    }


def sample_depth_validity(image, max_samples=2048):
    """Estimate usable depth ratio without numpy or decoding the full frame.

    The result is explicitly a sample ratio, not an exact full-image statistic.
    """
    encoding = str(getattr(image, "encoding", "") or "")
    data = getattr(image, "data", None)
    if data is None:
        return {"supported": False, "samples": 0, "valid": 0, "ratio": None}
    raw = bytes(data)
    item_size = 2 if encoding == "16UC1" else 4 if encoding == "32FC1" else 0
    if item_size == 0 or len(raw) < item_size:
        return {"supported": False, "samples": 0, "valid": 0, "ratio": None}

    count = len(raw) // item_size
    step = max(1, count // max(1, int(max_samples)))
    samples = 0
    valid = 0
    big = bool(getattr(image, "is_bigendian", 0))
    byteorder = "big" if big else "little"

    if encoding == "16UC1":
        for index in range(0, count, step):
            offset = index * 2
            value = int.from_bytes(raw[offset:offset + 2], byteorder=byteorder)
            samples += 1
            if value > 0:
                valid += 1
    else:
        import struct

        fmt = ">f" if big else "<f"
        for index in range(0, count, step):
            offset = index * 4
            value = struct.unpack_from(fmt, raw, offset)[0]
            samples += 1
            if math.isfinite(value) and value > 0.0:
                valid += 1

    ratio = valid / samples if samples else None
    return {
        "supported": True,
        "samples": samples,
        "valid": valid,
        "ratio": ratio,
    }


def inspect_object(localized_object, object_id=""):
    """Summarize one LocalizedObject using message truth only."""
    centroid = _point_tuple(getattr(localized_object, "centroid", None))
    dimensions = (
        float(getattr(localized_object, "length", 0.0) or 0.0),
        float(getattr(localized_object, "breadth", 0.0) or 0.0),
        float(getattr(localized_object, "height", 0.0) or 0.0),
    )
    cloud = getattr(localized_object, "segmented_pcl", None)
    cloud_points = int(getattr(cloud, "width", 0) or 0) * int(
        getattr(cloud, "height", 0) or 0
    )
    axis_norm = _vector_norm(getattr(localized_object, "axis", None))
    pose = getattr(localized_object, "pose", None)
    quaternion_norm = _quaternion_norm(getattr(pose, "orientation", None))

    centroid_ok = all(_finite(value) for value in centroid) and centroid[2] > 0.0
    dimensions_ok = all(_finite(value) and value > 0.0 for value in dimensions)
    axis_ok = _finite(axis_norm) and axis_norm > 1e-6
    pose_ok = _finite(quaternion_norm) and quaternion_norm > 1e-6

    if not centroid_ok:
        inspector_state = "invalid"
    elif dimensions_ok and axis_ok and pose_ok and cloud_points > 0:
        inspector_state = "valid"
    else:
        inspector_state = "degraded"

    return {
        "id": str(object_id or ""),
        "name": str(getattr(localized_object, "name", "") or ""),
        "centroid": centroid,
        "dimensions": dimensions,
        "cloud_points": cloud_points,
        "axis_norm": axis_norm,
        "quaternion_norm": quaternion_norm,
        "inspector_state": inspector_state,
    }


def summarize_p3_message(message, tracking=False):
    """Create a UI-safe dict from localization/tracking output."""
    objects = list(getattr(message, "objects", []) or [])
    ids = list(getattr(message, "object_ids", []) or []) if tracking else []
    summaries = []
    for index, obj in enumerate(objects):
        object_id = ids[index] if index < len(ids) else ""
        summaries.append(inspect_object(obj, object_id))

    frame = inspect_frame_contract(message)
    depth = sample_depth_validity(getattr(message, "depth_image", None))
    header = getattr(message, "header", None)
    return {
        "tracking": bool(tracking),
        "frame_id": str(getattr(header, "frame_id", "") or ""),
        "stamp_ns": _stamp_ns(getattr(header, "stamp", None)),
        "process_time_ms": int(getattr(message, "process_time", 0) or 0),
        "frame": frame,
        "depth": depth,
        "objects": summaries,
        "object_ids": [str(value) for value in ids],
        "lost_track_ids": [
            str(value) for value in (getattr(message, "lost_track_ids", []) or [])
        ],
    }


def diagnostics_values(message):
    """Extract the production inference-worker DiagnosticStatus values."""
    for status in getattr(message, "status", []) or []:
        if getattr(status, "name", "") != "easy_perception_deployment/inference_worker":
            continue
        return {
            str(getattr(item, "key", "")): str(getattr(item, "value", ""))
            for item in getattr(status, "values", []) or []
        }
    return {}


@dataclass
class InspectorEvent:
    kind: str
    payload: object


class P3InspectorWorker:
    """ROS subscriber worker with no Qt object access from its thread."""

    def __init__(self, events):
        self.events = events
        self._thread = None
        self._running = False
        self._context = None
        self._node = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="EPD3DInspector",
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        self._thread = None

    def _emit(self, kind, payload):
        self.events.put(InspectorEvent(kind, payload))

    def _run(self):
        try:
            import rclpy
            from diagnostic_msgs.msg import DiagnosticArray
            from epd_msgs.msg import EPDObjectLocalization, EPDObjectTracking
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy
        except ImportError as exc:
            self._emit("unavailable", f"ROS 2 Python messages unavailable: {exc}")
            return

        try:
            context = rclpy.context.Context()
            context.init(args=None)
            self._context = context
            node = Node("epd_3d_diagnostics_gui", context=context)
            self._node = node
            qos = QoSProfile(depth=10)
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
            node.create_subscription(
                EPDObjectLocalization,
                "/easy_perception_deployment/epd_localization_output",
                lambda msg: self._emit("p3", summarize_p3_message(msg, False)),
                qos,
            )
            node.create_subscription(
                EPDObjectTracking,
                "/easy_perception_deployment/epd_tracking_output",
                lambda msg: self._emit("p3", summarize_p3_message(msg, True)),
                qos,
            )
            node.create_subscription(
                DiagnosticArray,
                "/easy_perception_deployment/inference_diagnostics",
                lambda msg: self._emit("diagnostics", diagnostics_values(msg)),
                10,
            )
            self._emit("ready", "Listening for Localization/Tracking results.")
            while self._running and rclpy.ok(context=context):
                rclpy.spin_once(node, timeout_sec=0.1)
        except Exception as exc:
            self._emit("unavailable", f"3D inspector ROS error: {exc}")
        finally:
            try:
                if self._node is not None:
                    self._node.destroy_node()
                if self._context is not None:
                    self._context.shutdown()
            except Exception:
                pass
            self._node = None
            self._context = None


class ThreeDInspectorDialog(QDialog):
    """Live, read-only 3D diagnostics for Localization and Tracking."""

    def __init__(self, controller):
        super().__init__(controller.deploy)
        self.controller = controller
        self.setWindowTitle("EPD 3D Perception Inspector")
        self.resize(1080, 760)
        self.setMinimumSize(820, 600)

        outer = QVBoxLayout(self)
        title = QLabel("3D Perception Inspector", self)
        title.setObjectName("p3Title")
        subtitle = QLabel(
            "Inspect depth alignment, geometry quality and tracking IDs without changing the runtime.",
            self,
        )
        subtitle.setWordWrap(True)
        outer.addWidget(title)
        outer.addWidget(subtitle)

        self.state = QLabel("STOPPED", self)
        self.state.setObjectName("p3State")
        outer.addWidget(self.state, 0, Qt.AlignLeft)

        self.tabs = QTabWidget(self)
        outer.addWidget(self.tabs, 1)
        self._build_health_tab()
        self._build_objects_tab()
        self._build_tracks_tab()
        self._build_diagnostics_tab()

        footer = QHBoxLayout()
        self.refresh_button = QPushButton("Start / refresh", self)
        self.refresh_button.clicked.connect(controller.start)
        self.stop_button = QPushButton("Stop inspector", self)
        self.stop_button.clicked.connect(controller.stop)
        close = QPushButton("Close", self)
        close.clicked.connect(self.hide)
        footer.addWidget(self.refresh_button)
        footer.addWidget(self.stop_button)
        footer.addStretch(1)
        footer.addWidget(close)
        outer.addLayout(footer)
        self._style()

    def _build_health_tab(self):
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)
        self.health_summary = QLabel("No 3D result received yet.", tab)
        self.health_summary.setWordWrap(True)
        layout.addWidget(self.health_summary)

        grid = QGridLayout()
        labels = (
            "Frame / depth shape",
            "Camera intrinsics",
            "Result ↔ depth timestamp",
            "Depth encoding",
            "Sampled valid depth",
            "Last processing time",
            "Frame ID",
        )
        self.health_values = []
        for row, text in enumerate(labels):
            key = QLabel(text, tab)
            key.setObjectName("p3Key")
            value = QLabel("—", tab)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(key, row, 0)
            grid.addWidget(value, row, 1)
            self.health_values.append(value)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addStretch(1)

        note = QLabel(
            "Alignment here means the P3 result and embedded depth frame agree on dimensions and source timestamp. "
            "Use Camera Assistant for independent RGB/depth/CameraInfo topic health.",
            tab,
        )
        note.setWordWrap(True)
        note.setObjectName("p3Note")
        layout.addWidget(note)
        self.tabs.addTab(tab, "3D health")

    def _build_objects_tab(self):
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)
        self.object_table = QTableWidget(0, 9, tab)
        self.object_table.setHorizontalHeaderLabels(
            ["ID", "Class", "X m", "Y m", "Z m", "L×B×H m", "Cloud pts", "Axis |v|", "Inspector check"]
        )
        self.object_table.verticalHeader().setVisible(False)
        self.object_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.object_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.object_table, 1)
        note = QLabel(
            "Inspector check is a GUI-side sanity check of finite centroid, positive dimensions, axis, pose and point cloud. "
            "It is not a replacement for the production geometry counters.",
            tab,
        )
        note.setWordWrap(True)
        note.setObjectName("p3Note")
        layout.addWidget(note)
        self.tabs.addTab(tab, "Localized objects")

    def _build_tracks_tab(self):
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)
        self.track_summary = QTextBrowser(tab)
        layout.addWidget(self.track_summary, 1)
        self.tabs.addTab(tab, "Tracking IDs")

    def _build_diagnostics_tab(self):
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)
        self.geometry_summary = QLabel("Waiting for inference diagnostics…", tab)
        self.geometry_summary.setWordWrap(True)
        layout.addWidget(self.geometry_summary)
        self.diagnostics = QTextBrowser(tab)
        layout.addWidget(self.diagnostics, 1)
        filter_note = QLabel(
            "Plane/background filtering is intentionally not enabled by EPD-6. Add filtering only when workcell evidence shows a repeatable failure mode; diagnostics should reveal that evidence first.",
            tab,
        )
        filter_note.setWordWrap(True)
        filter_note.setObjectName("p3Note")
        layout.addWidget(filter_note)
        self.tabs.addTab(tab, "Geometry diagnostics")

    def update_p3(self, snapshot):
        frame = snapshot["frame"]
        depth = snapshot["depth"]
        state = frame["state"].upper()
        self.state.setText(f"LIVE • {state}")
        self.health_summary.setText(
            f"{len(snapshot['objects'])} object(s) • "
            f"{'Tracking' if snapshot['tracking'] else 'Localization'} result"
        )
        ratio = depth.get("ratio")
        ratio_text = "—" if ratio is None else f"{ratio * 100.0:.1f}% ({depth['valid']}/{depth['samples']} sampled)"
        shape_text = (
            f"{frame['frame_width']}×{frame['frame_height']} result • "
            f"{frame['depth_width']}×{frame['depth_height']} depth • "
            f"{'OK' if frame['shape_ok'] else 'MISMATCH'}"
        )
        values = (
            shape_text,
            "OK" if frame["intrinsics_ok"] else "INVALID",
            "Exact" if frame["stamps_match"] else "Mismatch / unavailable",
            f"{frame['depth_encoding'] or '—'} • {'OK' if frame['encoding_ok'] else 'unsupported'}",
            ratio_text,
            f"{snapshot['process_time_ms']} ms",
            snapshot["frame_id"] or "—",
        )
        for label, value in zip(self.health_values, values):
            label.setText(value)

        objects = snapshot["objects"]
        self.object_table.setRowCount(len(objects))
        for row, obj in enumerate(objects):
            centroid = obj["centroid"]
            dims = obj["dimensions"]
            cells = (
                obj["id"] or "—",
                obj["name"] or "—",
                _fmt(centroid[0]),
                _fmt(centroid[1]),
                _fmt(centroid[2]),
                " × ".join(_fmt(value) for value in dims),
                str(obj["cloud_points"]),
                _fmt(obj["axis_norm"]),
                obj["inspector_state"].upper(),
            )
            for column, value in enumerate(cells):
                self.object_table.setItem(row, column, QTableWidgetItem(value))
        self.object_table.resizeColumnsToContents()

        active = snapshot["object_ids"]
        lost = snapshot["lost_track_ids"]
        if snapshot["tracking"]:
            self.track_summary.setPlainText(
                "Current stable IDs:\n"
                + (", ".join(active) if active else "None in latest observation")
                + "\n\nLost in latest observation:\n"
                + (", ".join(lost) if lost else "None")
            )
        else:
            self.track_summary.setPlainText(
                "Latest result is Localization mode. Select Tracking to inspect persistent IDs and LOST transitions."
            )

    def update_diagnostics(self, metrics):
        if not metrics:
            return
        valid = _int_metric(metrics, "geometry_valid_total")
        degraded = _int_metric(metrics, "geometry_degraded_total")
        invalid = _int_metric(metrics, "geometry_invalid_total")
        self.geometry_summary.setText(
            f"Production geometry totals • valid {valid} • degraded {degraded} • invalid {invalid}"
        )
        lines = []
        for key in GEOMETRY_COUNTER_KEYS:
            if key in metrics:
                lines.append(f"{key}: {metrics[key]}")
        for key in (
            "confirmed_track_ids",
            "tracks_created",
            "tracks_lost",
            "associations_matched",
            "latest_observation_id",
            "last_completed_observation_id",
        ):
            if key in metrics:
                lines.append(f"{key}: {metrics[key]}")
        self.diagnostics.setPlainText("\n".join(lines) or "No geometry counters reported yet.")

    def set_unavailable(self, text):
        self.state.setText("UNAVAILABLE")
        self.health_summary.setText(str(text))

    def set_waiting(self, text="Waiting for Localization/Tracking output…"):
        self.state.setText("WAITING")
        self.health_summary.setText(text)

    def _style(self):
        self.setStyleSheet(
            """
            QDialog { background: #15191f; color: #e8edf2; }
            QLabel#p3Title { font-size: 24px; font-weight: 700; }
            QLabel#p3State { background: #29313a; border-radius: 6px; padding: 6px 10px; font-weight: 700; }
            QLabel#p3Key { color: #8f9aaa; font-weight: 600; }
            QLabel#p3Note { color: #8f9aaa; }
            QTableWidget, QTextBrowser { background: #11161b; border: 1px solid #303944; border-radius: 6px; }
            QPushButton { min-height: 32px; padding: 0 10px; }
            """
        )


class ThreeDInspectorController(QObject):
    """Attach the 3D Inspector to the existing Deploy window."""

    def __init__(self, main_window):
        super().__init__(main_window.deploy_window)
        self.main_window = main_window
        self.deploy = main_window.deploy_window
        self.events = queue.Queue()
        self.worker = P3InspectorWorker(self.events)
        self.dialog = ThreeDInspectorDialog(self)
        self.last_p3_at = None
        self.button = None
        self._install_button()
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(200)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.start()

    def _install_button(self):
        self.button = QPushButton("3D Inspector", self.deploy)
        self.button.setToolTip(
            "Inspect depth alignment, localized geometry and stable Tracking IDs."
        )
        self.button.clicked.connect(self.show)
        self.deploy.three_d_inspector_button = self.button
        ui = getattr(self.main_window, "_deploy_ui_controller", None)
        badge = getattr(ui, "header_badge", None)
        header = badge.parentWidget() if badge is not None else None
        if header is not None and header.layout() is not None:
            index = max(0, header.layout().count() - 1)
            header.layout().insertWidget(index, self.button, 0, Qt.AlignTop)
        elif self.deploy.layout() is not None:
            self.deploy.layout().addWidget(self.button)

    def show(self):
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        self.start()

    def start(self):
        mode = int(getattr(self.deploy, "usecase_mode", 0))
        if mode not in (3, 4):
            self.dialog.set_waiting(
                "3D Inspector is ready, but Deploy is not in Localization or Tracking mode. "
                "Select a P3 model and one of those modes to receive 3D results."
            )
        else:
            self.dialog.set_waiting()
        self.worker.start()

    def stop(self):
        self.worker.stop()
        self.dialog.state.setText("STOPPED")

    def shutdown(self):
        self.stop()
        self.poll_timer.stop()

    def _poll(self):
        processed = 0
        while processed < 20:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            processed += 1
            if event.kind == "p3":
                self.last_p3_at = time.monotonic()
                self.dialog.update_p3(event.payload)
            elif event.kind == "diagnostics":
                self.dialog.update_diagnostics(event.payload)
            elif event.kind == "unavailable":
                self.dialog.set_unavailable(event.payload)
            elif event.kind == "ready" and self.last_p3_at is None:
                self.dialog.set_waiting(str(event.payload))

        if self.last_p3_at is not None and time.monotonic() - self.last_p3_at > 3.0:
            if self.dialog.state.text().startswith("LIVE"):
                self.dialog.set_waiting("3D output is stale; waiting for a fresh Localization/Tracking result.")


def _fmt(value):
    return "—" if not _finite(value) else f"{float(value):.4f}"


def _int_metric(metrics, key):
    try:
        return int(float(metrics.get(key, 0)))
    except (TypeError, ValueError):
        return 0
