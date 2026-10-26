# Copyright 2026 Easy Perception Deployment contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Perception Monitor: the Deploy workspace for seeing what EPD detected.

Reorganises Deploy into a preview-first workspace and adds:

* camera-free replay of recorded perception sessions;
* the per-detection readiness table from
  ``/easy_perception_deployment/detection_readiness``;
* an Inspector for the selected detection and a Health panel.

Observability only: it does not change inference, thresholds, geometry
gates, ROS contracts, EMD or robot motion. ROS callbacks never touch Qt
objects; the worker hands parsed reports over through a bounded deque.
"""

from __future__ import annotations

import collections
import threading
import time
from pathlib import Path

from PySide6.QtCore import (
    QAbstractTableModel,
    QEvent,
    QModelIndex,
    QObject,
    QRectF,
    QSettings,
    QSize,
    Qt,
    QTimer,
)
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableView,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from windows.job_controller import JobState
from windows.perception_readiness import (
    DOWNSTREAM_NOTICE,
    READINESS_TOPIC,
    STATE_TEXT,
    STATES,
    UNAVAILABLE,
    ReplayPlayer,
    ReplayState,
    SessionError,
    format_stamp,
    format_stamp_short,
    live_health,
    load_session,
    parse_readiness_message,
    verify_image_bytes,
    reason_text,
    note_text,
    replay_health,
    roi_inside,
    short_reason,
    stale_after_s,
)

STATE_COLOURS = {
    "GRASP_CANDIDATE": "#4fbf7f",
    "POSE_VALID": "#3fb6c6",
    "LOCALISED": "#e0b84f",
    "DETECTED": "#8fa3c9",
    "REJECTED": "#e3646e",
}
HEALTH_STYLE = {
    "LIVE": "live",
    "REPLAY ACTIVE": "replay",
    "STARTING": "waiting",
    "WAITING": "waiting",
    "STALE": "waiting",
    "FAILED": "failed",
    "UNAVAILABLE": "failed",
    "STOPPED": "stopped",
}
SETTINGS_ORG = "WorkcellStudio"
SETTINGS_APP = "EasyPerceptionDeployment"
RECENT_KEY = "perception_monitor/recent_session"
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_SESSIONS = (
    ("Synthetic readiness states (test fixture)",
     _PACKAGE_ROOT / "fixtures" / "perception_sessions" /
     "synthetic_readiness_states"),
    ("Synthetic recording without readiness (test fixture)",
     _PACKAGE_ROOT / "fixtures" / "perception_sessions" /
     "synthetic_no_readiness"),
)


# --------------------------------------------------------------------------
# Live readiness topic worker
# --------------------------------------------------------------------------

class ReadinessTopicMonitor:
    """Background subscriber for the readiness topic.

    Uses a private ROS context and executor like the other Deploy workers.
    Parsed reports wait in a bounded deque; the GUI thread drains them.
    The context lives while Deploy is open; ``set_active`` only adds or
    removes the subscription, so switching modes creates no ROS context.
    """

    def __init__(self, max_pending=8):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._pending = collections.deque(maxlen=max_pending)
        self._arrivals = collections.deque(maxlen=20)
        self._publisher_count = 0
        self._error = ""
        self._active = True
        self._subscribed = False
        self.dropped = 0

    def set_active(self, active):
        with self._lock:
            if self._active != bool(active):
                self._active = bool(active)
                self._pending.clear()
                self._arrivals.clear()

    def subscribed(self):
        with self._lock:
            return self._subscribed

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        with self._lock:
            self._pending.clear()
            self._arrivals.clear()
            self._error = ""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="EPDReadinessMonitor")
        self._thread.start()

    def stop(self, timeout_sec=2.0):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_sec)
        stopped = thread is None or not thread.is_alive()
        self._thread = None
        return stopped

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def drain(self):
        with self._lock:
            reports = list(self._pending)
            self._pending.clear()
            return {
                "reports": reports,
                "arrivals": list(self._arrivals),
                "publisher_count": self._publisher_count,
                "error": self._error,
            }

    def _set_error(self, text):
        with self._lock:
            self._error = str(text)

    def _on_message(self, message):
        report = parse_readiness_message(message)
        report.received_monotonic = time.monotonic()
        with self._lock:
            if not self._active:
                return
            if len(self._pending) == self._pending.maxlen:
                self.dropped += 1
            self._pending.append(report)
            self._arrivals.append(report.received_monotonic)

    def _run(self):
        try:
            import rclpy
            from diagnostic_msgs.msg import DiagnosticArray
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import HistoryPolicy, QoSProfile
            from rclpy.qos import ReliabilityPolicy
        except ImportError:
            self._set_error(
                "ROS 2 Python is unavailable. Source ROS 2 and the EPD "
                "workspace before launching the GUI.")
            return
        context = node = executor = subscription = None
        try:
            context = rclpy.context.Context()
            context.init(args=None)
            node = rclpy.create_node(
                "epd_perception_monitor", context=context)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            # Matches the producer: reliable, keep-last 10.
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST, depth=10,
                reliability=ReliabilityPolicy.RELIABLE)
            next_count = 0.0
            while not self._stop.is_set() and context.ok():
                with self._lock:
                    active = self._active
                if active and subscription is None:
                    subscription = node.create_subscription(
                        DiagnosticArray, READINESS_TOPIC, self._on_message,
                        qos)
                elif not active and subscription is not None:
                    node.destroy_subscription(subscription)
                    subscription = None
                with self._lock:
                    self._subscribed = subscription is not None
                executor.spin_once(timeout_sec=0.1)
                now = time.monotonic()
                if now >= next_count:
                    count = node.count_publishers(READINESS_TOPIC)
                    with self._lock:
                        self._publisher_count = count
                    next_count = now + 1.0
        except Exception as exc:  # noqa: BLE001 - shown to the operator
            self._set_error(f"Readiness monitor failed: {exc}")
        finally:
            with self._lock:
                self._subscribed = False
            if executor is not None:
                try:
                    executor.shutdown(timeout_sec=0.5)
                except Exception:  # noqa: BLE001
                    pass
            if node is not None:
                try:
                    node.destroy_node()
                except Exception:  # noqa: BLE001
                    pass
            if context is not None:
                try:
                    context.shutdown()
                except Exception:  # noqa: BLE001
                    pass


# --------------------------------------------------------------------------
# Widgets
# --------------------------------------------------------------------------

class PerceptionView(QWidget):
    """Frame view that draws ROIs only for a stamp-matched report."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("perceptionView")
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image = None
        self._records = []
        self._draw_rois = False
        self._selected_key = None
        self._placeholder = "Open a recording to replay perception."
        self._notice = ""
        self._stale = False

    def show_frame(self, image, records, draw_rois, notice=""):
        self._image = image
        self._records = list(records or [])
        self._draw_rois = bool(draw_rois)
        self._notice = notice
        self.update()

    def show_placeholder(self, text):
        self._image = None
        self._records = []
        self._draw_rois = False
        self._notice = ""
        self._placeholder = text
        self.update()

    def set_selected(self, key):
        self._selected_key = key
        self.update()

    def set_stale(self, stale):
        self._stale = bool(stale)
        self.update()

    def draws_rois(self):
        return self._draw_rois and self._image is not None

    def image_rect(self):
        if self._image is None or self._image.isNull():
            return QRectF()
        area = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        scale = min(area.width() / self._image.width(),
                    area.height() / self._image.height())
        width = self._image.width() * scale
        height = self._image.height() * scale
        return QRectF(area.x() + (area.width() - width) / 2,
                      area.y() + (area.height() - height) / 2,
                      width, height)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#090c11"))
        if self._image is None or self._image.isNull():
            painter.setPen(QColor("#8d99ab"))
            painter.drawText(self.rect().adjusted(16, 16, -16, -16),
                             Qt.AlignCenter | Qt.TextWordWrap,
                             self._placeholder)
            return
        target = self.image_rect()
        painter.drawImage(target, self._image)
        if self._draw_rois:
            self._paint_rois(painter, target)
        if self._stale:
            painter.fillRect(target, QColor(9, 12, 17, 110))
            self._paint_tag(painter, target, "REPLAY ENDED · NOT UPDATING",
                            QColor("#e8c98f"), top=True)
        if self._notice:
            self._paint_tag(painter, target, self._notice,
                            QColor("#e3a1a1"), top=False)

    def _paint_rois(self, painter, target):
        scale = target.width() / self._image.width()
        font = QFont(painter.font())
        font.setPixelSize(11)
        font.setBold(True)
        painter.setFont(font)
        for record in self._records:
            if not roi_inside(record.roi, self._image.width(),
                              self._image.height()):
                continue
            x, y, w, h = record.roi
            rect = QRectF(target.x() + x * scale, target.y() + y * scale,
                          w * scale, h * scale)
            colour = QColor(STATE_COLOURS.get(record.state, "#cccccc"))
            selected = record.key == self._selected_key
            if selected:
                fill = QColor(colour)
                fill.setAlpha(60)
                painter.fillRect(rect, fill)
            painter.setPen(QPen(colour, 3.0 if selected else 1.4))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)
            label = f"#{record.index} {record.class_name}"
            metrics = painter.fontMetrics()
            box = QRectF(rect.x(), rect.y() - 16,
                         metrics.horizontalAdvance(label) + 8, 16)
            if box.y() < target.y():
                box.moveTop(rect.y())
            backdrop = colour if selected else QColor(9, 12, 17, 190)
            painter.fillRect(box, backdrop)
            painter.setPen(QColor("#0b0f14") if selected else colour)
            painter.drawText(box, Qt.AlignCenter, label)

    def _paint_tag(self, painter, target, text, colour, top):
        font = QFont(painter.font())
        font.setPixelSize(11)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = min(target.width() - 12,
                    metrics.horizontalAdvance(text) + 18)
        height = 22 if metrics.horizontalAdvance(text) + 18 <= width else 38
        y = target.y() + 6 if top else target.bottom() - height - 6
        box = QRectF(target.x() + 6, y, width, height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(9, 12, 17, 225)))
        painter.drawRoundedRect(box, 6, 6)
        painter.setPen(colour)
        painter.drawText(box.adjusted(8, 0, -8, 0),
                         Qt.AlignVCenter | Qt.AlignLeft | Qt.TextWordWrap,
                         text)


class _ImageCache:
    """Small bounded cache so replay never buffers a whole recording.

    Each frame is read once: the same bytes are hashed against the recording
    and decoded, so a modified frame is never shown.
    """

    def __init__(self, capacity=4):
        self._capacity = capacity
        self._images = collections.OrderedDict()

    def get(self, frame):
        key = str(frame.image_path)
        if key in self._images:
            self._images.move_to_end(key)
            return self._images[key]
        try:
            data = Path(key).read_bytes()
        except OSError:
            return None
        verify_image_bytes(frame, data)
        image = QImage.fromData(data)
        if image.isNull():
            return None
        self._images[key] = image
        while len(self._images) > self._capacity:
            self._images.popitem(last=False)
        return image

    def clear(self):
        self._images.clear()

    def __len__(self):
        return len(self._images)


SELECT_HINT = "Select a detection in the table to see its evidence."


def _value_label(parent):
    label = QLabel(UNAVAILABLE, parent)
    label.setObjectName("monitorValue")
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    return label


def usable_3d_text(record):
    """One-line answer to 'is the 3D information usable?' from the report."""
    position = "valid" if record.centroid_m is not None else "not valid"
    orientation = "valid" if record.axis is not None else "not valid"
    if record.state == "REJECTED":
        return "3D information: not usable (2D evidence rejected)"
    return (f"3D information: position {position} · orientation "
            f"{orientation}")


def _fmt_vector(values, digits=3):
    return ", ".join(f"{value:.{digits}f}" for value in values)


class ObjectInspector(QScrollArea):
    """Evidence for one detection, exactly as the report states it."""

    FIELDS = (
        ("object", "Object"),
        ("confidence", "Confidence"),
        ("proposal", "Detector proposal"),
        ("track", "Track ID"),
        ("readiness", "Readiness"),
        ("downstream", "Downstream"),
        ("reasons", "Why"),
        ("box", "Bounding box"),
        ("mask", "Mask"),
        ("depth", "Depth"),
        ("position", "3D position"),
        ("dimensions", "Dimensions"),
        ("orientation", "Orientation"),
        ("source", "Source frame"),
        ("association", "Image match"),
        ("issues", "Report check"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("objectInspector")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        # A constant viewport width keeps wrapped values from being clipped
        # when a scrollbar would otherwise appear after layout.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        body = QWidget(self)
        body.setObjectName("monitorPanelBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        self.title = QLabel("No detection selected", body)
        self.title.setObjectName("inspectorTitle")
        self.title.setWordWrap(True)
        self.hint = QLabel(SELECT_HINT, body)
        self.hint.setObjectName("monitorHint")
        self.hint.setWordWrap(True)
        self.usable = QLabel("", body)
        self.usable.setObjectName("inspectorUsable")
        self.usable.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.usable)
        layout.addWidget(self.hint)
        self.form_host = QWidget(body)
        form = QFormLayout(self.form_host)
        form.setContentsMargins(0, 4, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(7)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.values = {}
        for key, text in self.FIELDS:
            name = QLabel(text, self.form_host)
            name.setObjectName("monitorField")
            value = _value_label(self.form_host)
            form.addRow(name, value)
            self.values[key] = value
        layout.addWidget(self.form_host)
        layout.addStretch(1)
        self.setWidget(body)
        self.clear()

    def clear(self, hint=SELECT_HINT):
        self.record = None
        self.title.setText("No detection selected")
        self.usable.setText("")
        self.usable.hide()
        self.hint.setText(hint)
        self.hint.show()
        self.form_host.hide()
        for value in self.values.values():
            value.setText(UNAVAILABLE)

    def show_record(self, record, report, association_text):
        self.record = record
        self.form_host.show()
        self.hint.hide()
        name = record.class_name or "unnamed class"
        self.title.setText(f"#{record.index} · {name}")
        self.usable.setText(usable_3d_text(record))
        self.usable.show()
        v = self.values
        v["object"].setText(name if record.class_name else UNAVAILABLE)
        if record.confidence is None:
            v["confidence"].setText(UNAVAILABLE)
        else:
            v["confidence"].setText(
                f"{record.confidence:.3f} (detector score, reported "
                f"'{record.confidence_text}')")
        v["proposal"].setText(
            f"#{record.index} of this report ({record.status_name})")
        v["track"].setText(
            record.track_id if record.track_id else
            "Unavailable (no tracker assignment in this report)")
        label, meaning = STATE_TEXT.get(
            record.state, (record.state or "Unknown", ""))
        v["readiness"].setText(
            f"<b style='color:{STATE_COLOURS.get(record.state, '#ccc')}'>"
            f"{record.state or 'UNKNOWN'}</b> · {label}<br>{meaning}")
        v["downstream"].setText(DOWNSTREAM_NOTICE)
        lines = [f"{reason_text(code)} <code>{code}</code>"
                 for code in record.reasons]
        if record.note:
            lines.append(f"{note_text(record.note)} "
                         f"<code>{record.note}</code>")
        v["reasons"].setText(
            "<br>".join(lines) if lines else "No failure reported.")
        if record.roi is None:
            v["box"].setText(UNAVAILABLE)
        else:
            x, y, w, h = record.roi
            v["box"].setText(f"x {x}, y {y}, {w} × {h} px")
        if record.mask_pixels is None:
            v["mask"].setText(UNAVAILABLE)
        elif record.mask_pixels == 0:
            v["mask"].setText("No owned mask pixels")
        else:
            v["mask"].setText(
                f"Owned mask, {record.mask_pixels:,} px (the mask image is "
                "not part of the readiness report)")
        if record.valid_depth_pixels is None:
            v["depth"].setText(UNAVAILABLE)
        elif record.mask_pixels:
            share = 100.0 * record.valid_depth_pixels / record.mask_pixels
            v["depth"].setText(
                f"{record.valid_depth_pixels:,} valid depth px reported in "
                f"the mask ({share:.0f}%)")
        else:
            v["depth"].setText(
                f"{record.valid_depth_pixels:,} valid depth px reported")
        frame = report.frame_id or "unknown frame"
        if record.centroid_m is not None:
            v["position"].setText(
                f"Valid · centroid {_fmt_vector(record.centroid_m)} m "
                f"in {frame}")
        elif record.position_valid is False:
            v["position"].setText("Not valid (no measured 3D position)")
        else:
            v["position"].setText(UNAVAILABLE)
        v["dimensions"].setText(
            "Unavailable (not part of the readiness report)")
        if record.axis is not None:
            v["orientation"].setText(
                f"Valid · principal axis {_fmt_vector(record.axis)}")
        elif record.orientation_valid is False:
            v["orientation"].setText("Not valid")
        else:
            v["orientation"].setText(UNAVAILABLE)
        observation = record.observation_id or UNAVAILABLE
        hardware = record.hardware_id or UNAVAILABLE
        v["source"].setText(
            f"{frame} · stamp {format_stamp(report.stamp_ns)}<br>"
            f"observation {observation} · camera {hardware}")
        v["association"].setText(association_text)
        issues = list(record.issues) + list(report.issues)
        v["issues"].setText(
            "<br>".join(issues) if issues else
            "Consistent with the readiness contract.")


class HealthPanel(QScrollArea):
    """Session, stream and pipeline health in operator language."""

    ROWS = (
        ("source", "Source"),
        ("session", "Session"),
        ("readiness", "Readiness stream"),
        ("frames", "RGB frames"),
        ("rate", "Report rate"),
        ("latency", "Latency"),
        ("age", "Observation age"),
        ("objects", "Objects"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("healthPanel")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        body = QWidget(self)
        body.setObjectName("monitorPanelBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        top = QHBoxLayout()
        self.badge = QLabel("STOPPED", body)
        self.badge.setObjectName("monitorHealthBadge")
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setMinimumWidth(120)
        self.headline = QLabel("", body)
        self.headline.setObjectName("monitorHeadline")
        self.headline.setWordWrap(True)
        top.addWidget(self.badge, 0, Qt.AlignTop)
        top.addWidget(self.headline, 1)
        layout.addLayout(top)
        form_host = QWidget(body)
        form = QFormLayout(form_host)
        form.setContentsMargins(0, 4, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(7)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.values = {}
        for key, text in self.ROWS:
            name = QLabel(text, form_host)
            name.setObjectName("monitorField")
            value = _value_label(form_host)
            value.setText("—")
            form.addRow(name, value)
            self.values[key] = value
        layout.addWidget(form_host)
        title = QLabel("MESSAGES", body)
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.messages = QLabel("No problems reported.", body)
        self.messages.setObjectName("monitorMessages")
        self.messages.setWordWrap(True)
        self.messages.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.messages)
        layout.addStretch(1)
        self.setWidget(body)

    def set_health(self, health):
        _set_badge(self.badge, health.state)
        self.headline.setText(health.headline)
        if health.messages:
            self.messages.setText(
                "<br>".join(f"• {message}" for message in health.messages))
        else:
            self.messages.setText("No problems reported.")
        # Amber only when there is something to read; "no problems" is not a warning.
        self.messages.setProperty("state", "warn" if health.messages else "ok")
        self.messages.style().unpolish(self.messages)
        self.messages.style().polish(self.messages)

    def set_value(self, key, text):
        self.values[key].setText(text)


def _set_badge(label, state):
    label.setText(state)
    label.setProperty("monitorState", HEALTH_STYLE.get(state, "stopped"))
    label.style().unpolish(label)
    label.style().polish(label)


TABLE_COLUMNS = (
    "#", "Object", "Confidence", "Track", "Readiness", "Reason",
    "Centroid (m)", "Source stamp", "Age",
)
AGE_COLUMN = 8


class ReadinessTableModel(QAbstractTableModel):
    """Read-only rows for one readiness report.

    A model rather than QTableWidgetItems: nothing per cell is a wrapped C++
    object, so refreshing every frame allocates no Qt objects.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []
        self._age = "—"

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(TABLE_COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return TABLE_COLUMNS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._rows):
            return None
        row = self._rows[index.row()]
        column = index.column()
        if role == Qt.DisplayRole:
            return self._age if column == AGE_COLUMN else row["cells"][column]
        if role == Qt.ToolTipRole:
            return row["tips"][column]
        if column == 4 and role == Qt.ForegroundRole:
            return QBrush(QColor(row["colour"]))
        if column == 4 and role == Qt.FontRole:
            font = QFont()
            font.setBold(True)
            return font
        if column in (0, 2, 3) and role == Qt.TextAlignmentRole:
            return int(Qt.AlignCenter)
        return None

    def set_report(self, report):
        self.beginResetModel()
        self._rows = [] if report is None else [
            _row_for(record, report) for record in report.records]
        self.endResetModel()

    def set_age(self, text):
        if text == self._age:
            return
        self._age = text
        if self._rows:
            self.dataChanged.emit(
                self.index(0, AGE_COLUMN),
                self.index(len(self._rows) - 1, AGE_COLUMN),
                [Qt.DisplayRole])

    def text(self, row, column):
        return self.data(self.index(row, column), Qt.DisplayRole)

    def tooltip(self, row, column):
        return self.data(self.index(row, column), Qt.ToolTipRole)


def _row_for(record, report):
    confidence = ("—" if record.confidence is None else
                  f"{record.confidence:.2f}")
    centroid = ("—" if record.centroid_m is None else
                _fmt_vector(record.centroid_m))
    cells = (
        str(record.index), record.class_name or "—", confidence,
        record.track_id or "—", record.state or "UNKNOWN",
        short_reason(record) or "—", centroid,
        format_stamp_short(report.stamp_ns), "—",
    )
    tips = (
        record.status_name,
        record.class_name,
        ("Detector confidence as reported: " + record.confidence_text
         if record.confidence_text else UNAVAILABLE),
        ("Track ID" if record.track_id else
         "Unavailable: no tracker assignment"),
        " ".join(STATE_TEXT.get(record.state, ("", ""))),
        "\n".join(record.reason_lines()) or "No failure reported.",
        ("Measured centroid in " + report.frame_id
         if record.centroid_m else UNAVAILABLE),
        format_stamp(report.stamp_ns),
        "Time since this report (replay clock in replay mode)",
    )
    return {"cells": cells, "tips": tips,
            "colour": STATE_COLOURS.get(record.state, "#cccccc")}


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------

class PerceptionMonitorController(QObject):
    """Builds the workspace and owns replay, live monitoring and selection."""

    MODE_LIVE = "live"
    MODE_REPLAY = "replay"

    def __init__(self, main_window, clock=time.monotonic):
        super().__init__(main_window.deploy_window)
        self.main_window = main_window
        self.deploy = main_window.deploy_window
        self._clock = clock
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP, self)
        self.mode = self.MODE_LIVE
        self.session = None
        self.load_error = ""
        self.player = ReplayPlayer(clock=clock)
        self.live_monitor = ReadinessTopicMonitor()
        self.cache = _ImageCache()
        self.current_report = None
        self.current_frame = None
        self.current_records = []
        self.selected_key = None
        self._live_report_at = None
        self._live_arrivals = []
        self._live_publisher_count = 0
        self._live_error = ""
        self._running_since = None
        self._shown_index = -1
        self._is_shut_down = False
        self._programmatic_selection = False
        self._refreshing = False
        self.health_state = None

        self.replay_timer = QTimer(self)
        self.replay_timer.setInterval(50)
        self.replay_timer.timeout.connect(self._replay_tick)
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(200)
        self.live_timer.timeout.connect(self._live_tick)

        self._build_workspace()
        self._append_style()
        # The ROS worker is plain Python: stop it even if Deploy is deleted
        # without a close event. The callback must not reference self.
        live_monitor = self.live_monitor
        self.destroyed.connect(lambda *_: live_monitor.stop())
        self.deploy._job_controller.state_changed.connect(self._on_job_state)
        self.deploy.installEventFilter(self)
        self._set_mode(self.MODE_LIVE)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    # ---------------------------------------------------------------- layout
    def _build_workspace(self):
        deploy = self.deploy
        root = deploy.layout()
        scroll = deploy.findChild(QScrollArea, "deployScroll")
        content = deploy.findChild(QWidget, "deployContent")
        self.preview_card = getattr(deploy, "_epd2_preview_card", None)

        self.workspace = QWidget(deploy)
        self.workspace.setObjectName("perceptionWorkspace")
        outer = QVBoxLayout(self.workspace)
        outer.setContentsMargins(14, 10, 14, 8)
        outer.setSpacing(8)
        outer.addWidget(self._build_transport_bar())

        self.main_splitter = QSplitter(Qt.Horizontal, self.workspace)
        self.main_splitter.setObjectName("monitorMainSplitter")
        self.main_splitter.setChildrenCollapsible(False)
        self.left_splitter = QSplitter(Qt.Vertical, self.main_splitter)
        self.left_splitter.setObjectName("monitorLeftSplitter")
        self.left_splitter.setChildrenCollapsible(False)

        self.view_stack = QStackedWidget(self.left_splitter)
        self.view_stack.setObjectName("monitorViewStack")
        if self.preview_card is not None:
            self._reflow_setup(content, exclude=self.preview_card)
            self.view_stack.addWidget(self.preview_card)
            surface = getattr(self._live_controller(), "surface", None)
            if surface is not None:
                surface.setMinimumSize(320, 160)
        else:
            self._reflow_setup(content, exclude=None)
            placeholder = QLabel("Live preview unavailable.", self.view_stack)
            self.view_stack.addWidget(placeholder)
        self.view_stack.addWidget(self._build_replay_card())
        self.left_splitter.addWidget(self.view_stack)
        self.left_splitter.addWidget(self._build_detections_card())
        self.left_splitter.setStretchFactor(0, 3)
        self.left_splitter.setStretchFactor(1, 2)

        self.side_tabs = QTabWidget(self.main_splitter)
        self.side_tabs.setObjectName("monitorSideTabs")
        self.side_tabs.setMinimumWidth(340)
        self.inspector = ObjectInspector(self.side_tabs)
        self.health = HealthPanel(self.side_tabs)
        self.side_tabs.addTab(self.inspector, "Inspector")
        self.side_tabs.addTab(self.health, "Health")
        if scroll is not None:
            root.removeWidget(scroll)
            scroll.setParent(self.side_tabs)
            self.side_tabs.addTab(scroll, "Setup")
            self.setup_scroll = scroll
        else:
            self.setup_scroll = None
        self.side_tabs.setCurrentIndex(self.side_tabs.count() - 1)
        self.main_splitter.addWidget(self.left_splitter)
        self.main_splitter.addWidget(self.side_tabs)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setSizes([900, 400])
        self.left_splitter.setSizes([360, 230])
        outer.addWidget(self.main_splitter, 1)

        root.addWidget(self.workspace, 1, 0)
        root.setRowStretch(1, 1)
        subtitle = deploy.findChild(QLabel, "deploySubtitle")
        if subtitle is not None:
            subtitle.setText(
                "Configure and run perception, or replay a recording, then "
                "inspect what EPD detected.")
        deploy.perception_monitor = self

    def _live_controller(self):
        return getattr(self.main_window, "_epd2_live_perception_controller",
                       None)

    @staticmethod
    def _reflow_setup(content, exclude):
        """Re-stack the setup cards in one column for the side panel."""
        if content is None or content.layout() is None:
            return
        grid = content.layout()
        placed = []
        for index in range(grid.count()):
            item = grid.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is None:
                continue
            row, column, _rows, _columns = grid.getItemPosition(index)
            placed.append((row, column, widget))
        for _row, _column, widget in placed:
            grid.removeWidget(widget)
        row = 0
        for _r, _c, widget in sorted(placed, key=lambda t: (t[0], t[1])):
            if widget is exclude:
                continue
            grid.addWidget(widget, row, 0, 1, 1)
            row += 1
        for column in range(1, max(grid.columnCount(), 2)):
            grid.setColumnStretch(column, 0)
        grid.setColumnStretch(0, 1)
        grid.setRowStretch(row, 1)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setVerticalSpacing(10)

    def _build_transport_bar(self):
        bar = QFrame(self.workspace)
        bar.setObjectName("monitorTransport")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(6)

        self.live_button = QPushButton("Live camera", bar)
        self.replay_button = QPushButton("Replay recording", bar)
        self.mode_group = QButtonGroup(bar)
        for button in (self.live_button, self.replay_button):
            button.setCheckable(True)
            button.setObjectName("monitorModeButton")
            self.mode_group.addButton(button)
            layout.addWidget(button)
        self.live_button.setToolTip(
            "Show the configured camera and live EPD readiness output.")
        self.replay_button.setToolTip(
            "Replay a recorded perception session. No camera or inference "
            "is needed.")
        self.live_button.clicked.connect(
            lambda: self._set_mode(self.MODE_LIVE))
        self.replay_button.clicked.connect(
            lambda: self._set_mode(self.MODE_REPLAY))

        layout.addSpacing(8)
        self.open_button = QToolButton(bar)
        self.open_button.setObjectName("monitorOpenButton")
        self.open_button.setText("Open recording…")
        self.open_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.open_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.open_menu = QMenu(self.open_button)
        self.open_menu.aboutToShow.connect(self._populate_open_menu)
        self.open_button.setMenu(self.open_menu)
        self.open_button.clicked.connect(self.choose_session)
        layout.addWidget(self.open_button)

        self.play_button = QPushButton("Play", bar)
        self.pause_button = QPushButton("Pause", bar)
        self.restart_button = QPushButton("Restart", bar)
        self.stop_button = QPushButton("Stop", bar)
        self.prev_button = QPushButton("◀", bar)
        self.next_button = QPushButton("▶", bar)
        self.prev_button.setToolTip("Previous frame (while paused)")
        self.next_button.setToolTip("Next frame (while paused)")
        for button in (self.play_button, self.pause_button,
                       self.restart_button, self.stop_button,
                       self.prev_button, self.next_button):
            button.setObjectName("monitorTransportButton")
            layout.addWidget(button)
        self.prev_button.setFixedWidth(34)
        self.next_button.setFixedWidth(34)
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.restart_button.clicked.connect(self.restart)
        self.stop_button.clicked.connect(self.stop)
        self.prev_button.clicked.connect(lambda: self.step(-1))
        self.next_button.clicked.connect(lambda: self.step(1))

        self.position_label = QLabel("", bar)
        self.position_label.setObjectName("monitorPosition")
        self.position_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.position_label, 1)
        self.health_badge = QLabel("STOPPED", bar)
        self.health_badge.setObjectName("monitorHealthBadge")
        self.health_badge.setAlignment(Qt.AlignCenter)
        self.health_badge.setMinimumWidth(118)
        layout.addWidget(self.health_badge)
        return bar

    def _build_replay_card(self):
        card = QFrame(self.view_stack)
        card.setObjectName("sectionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)
        header = QHBoxLayout()
        self.provenance_label = QLabel("NO RECORDING", card)
        self.provenance_label.setObjectName("monitorProvenance")
        self.session_label = QLabel("", card)
        self.session_label.setObjectName("sectionSubtitle")
        self.session_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred)
        header.addWidget(self.provenance_label, 0)
        header.addWidget(self.session_label, 1)
        layout.addLayout(header)
        self.view = PerceptionView(card)
        layout.addWidget(self.view, 1)
        self.frame_label = QLabel("", card)
        self.frame_label.setObjectName("monitorHint")
        self.frame_label.setWordWrap(True)
        layout.addWidget(self.frame_label)
        return card

    def _build_detections_card(self):
        card = QFrame(self.left_splitter)
        card.setObjectName("sectionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)
        header = QHBoxLayout()
        title = QLabel("DETECTIONS", card)
        title.setObjectName("sectionTitle")
        self.summary_label = QLabel("", card)
        self.summary_label.setObjectName("sectionSubtitle")
        self.summary_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred)
        header.addWidget(title)
        header.addWidget(self.summary_label, 1)
        layout.addLayout(header)
        self.table_stack = QStackedWidget(card)
        self.table_model = ReadinessTableModel(self)
        self.table = QTableView(self.table_stack)
        self.table.setObjectName("readinessTable")
        self.table.setModel(self.table_model)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.setMinimumHeight(110)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setAlternatingRowColors(True)
        header_view = self.table.horizontalHeader()
        header_view.setMinimumSectionSize(36)
        for column in range(len(TABLE_COLUMNS)):
            header_view.setSectionResizeMode(
                column, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.selectionModel().selectionChanged.connect(
            self._on_table_selection)
        self.unavailable_label = QLabel("", self.table_stack)
        self.unavailable_label.setObjectName("readinessUnavailable")
        self.unavailable_label.setAlignment(Qt.AlignCenter)
        self.unavailable_label.setWordWrap(True)
        self.table_stack.addWidget(self.table)
        self.table_stack.addWidget(self.unavailable_label)
        layout.addWidget(self.table_stack, 1)
        return card

    # ----------------------------------------------------------------- modes
    def _set_mode(self, mode):
        self.mode = mode
        live = mode == self.MODE_LIVE
        self.live_button.setChecked(live)
        self.replay_button.setChecked(not live)
        for widget in (self.open_button, self.play_button,
                       self.pause_button, self.restart_button,
                       self.stop_button, self.prev_button,
                       self.next_button):
            widget.setVisible(not live)
        self.view_stack.setCurrentIndex(0 if live else 1)
        self._clear_selection()
        if live:
            self.stop()
            self._reset_live_state()
            self._start_live_preview()
            self._start_live_monitor()
            self._render_live(force=True)
        else:
            self._stop_live_monitor()
            self._stop_live_preview()
            self._render_replay(force=True)
        self._update_controls()

    def _start_live_preview(self):
        controller = self._live_controller()
        if controller is not None and self.deploy.isVisible():
            controller._start_monitor()

    def _stop_live_preview(self):
        controller = self._live_controller()
        if controller is not None:
            controller._stop_monitor()

    def _reset_live_state(self):
        self.current_report = None
        self.current_records = []
        self._live_report_at = None
        self._live_arrivals = []

    def _start_live_monitor(self):
        if self._is_shut_down or not self.deploy.isVisible():
            return
        self._reset_live_state()
        self.live_monitor.set_active(True)
        self.live_monitor.start()
        self.live_timer.start()

    def _stop_live_monitor(self):
        """Leave live mode: drop the subscription, keep the ROS context."""
        self.live_timer.stop()
        self.live_monitor.set_active(False)

    # ---------------------------------------------------------------- replay
    def _populate_open_menu(self):
        self.open_menu.clear()
        choose = self.open_menu.addAction("Open recording folder…")
        choose.triggered.connect(self.choose_session)
        recent = self.settings.value(RECENT_KEY, "")
        if recent and Path(str(recent)).exists():
            action = self.open_menu.addAction(
                f"Recent: {Path(str(recent)).name}")
            action.triggered.connect(
                lambda _=False, path=str(recent): self.open_session(path))
        self.open_menu.addSeparator()
        for title, path in BUNDLED_SESSIONS:
            if path.exists():
                action = self.open_menu.addAction(title)
                action.triggered.connect(
                    lambda _=False, p=str(path): self.open_session(p))

    def choose_session(self):
        start = str(self.settings.value(RECENT_KEY, "") or
                    BUNDLED_SESSIONS[0][1])
        directory = QFileDialog.getExistingDirectory(
            self.deploy, "Open perception recording", start)
        if directory:
            self.open_session(directory)

    def open_session(self, path):
        """Load a recording; return True when it is ready to play."""
        if self.mode != self.MODE_REPLAY:
            self._set_mode(self.MODE_REPLAY)
        self.stop()
        self.player.unload()
        self.cache.clear()
        self.session = None
        self._clear_selection()
        try:
            # Image hashes are checked frame by frame as they are first
            # shown, so a long recording opens without freezing the GUI.
            session = load_session(path, verify_images=False)
        except SessionError as exc:
            self.load_error = str(exc)
            self._render_replay(force=True)
            self._update_controls()
            return False
        self.load_error = ""
        self.session = session
        self.player.load([frame.stamp_ns for frame in session.frames])
        if not session.is_synthetic:
            self.settings.setValue(RECENT_KEY, str(session.path))
        self._render_replay(force=True)
        self._update_controls()
        return True

    def _fail_session(self, message):
        """Stop replay of a recording that failed verification."""
        self.replay_timer.stop()
        self.player.unload()
        self.cache.clear()
        self.session = None
        self.load_error = message
        self._clear_selection()
        self._render_replay(force=True)
        self._update_controls()

    def play(self):
        if self.session is None:
            return
        self.player.play()
        self.replay_timer.start()
        self._replay_tick()

    def pause(self):
        self.player.pause()
        self.replay_timer.stop()
        self._render_replay(force=True)
        self._update_controls()

    def restart(self):
        if self.session is None:
            return
        self._clear_selection()
        self.player.restart()
        self.replay_timer.start()
        self._replay_tick(force=True)

    def stop(self):
        self.replay_timer.stop()
        self.player.stop()
        self.cache.clear()
        self._clear_selection()
        self._render_replay(force=True)
        self._update_controls()

    def step(self, delta):
        self.player.step(delta)
        self._render_replay(force=True)
        self._update_controls()

    def _replay_tick(self, force=False):
        self.player.tick()
        if self.player.state != ReplayState.PLAYING:
            self.replay_timer.stop()
        self._render_replay(force=force)
        self._update_controls()

    def _render_replay(self, force=False):
        if self.mode != self.MODE_REPLAY:
            return
        session = self.session
        state = self.player.state
        index = self.player.index
        frame = None
        if session is not None and index is not None:
            frame = session.frames[index]
        changed = force or index != self._shown_index
        self._shown_index = -1 if index is None else index
        if session is None:
            self.provenance_label.setText("NO RECORDING")
            self.provenance_label.setProperty("provenance", "none")
            self.session_label.setText(
                self.load_error or "Open a recording or a bundled test "
                "fixture to replay perception without a camera.")
        else:
            synthetic = session.is_synthetic
            self.provenance_label.setText(
                "SYNTHETIC TEST FIXTURE · NOT CAMERA DATA" if synthetic
                else "RECORDED REPLAY · NOT A LIVE CAMERA")
            self.provenance_label.setProperty(
                "provenance", "synthetic" if synthetic else "recorded")
            self.session_label.setText(session.name)
            self.session_label.setToolTip(
                session.provenance.get("summary", ""))
        self.provenance_label.style().unpolish(self.provenance_label)
        self.provenance_label.style().polish(self.provenance_label)
        if changed:
            self._show_replay_frame(session, frame, state)
            if self.session is not session:
                return  # the frame failed verification and was re-rendered
        self.view.set_stale(state == ReplayState.ENDED)
        self._render_replay_health(session, frame, state)
        self._update_ages()

    def _show_replay_frame(self, session, frame, state):
        self.current_frame = frame
        if session is None:
            self.view.show_placeholder(
                self.load_error or "Open a recording to replay perception. "
                "No camera is needed.")
            self.frame_label.setText("")
            self._show_report(None, unavailable_reason=(
                "No recording loaded." if not self.load_error else
                "Recording not loaded."))
            return
        if frame is None:
            text = ("Replay stopped. No detections are shown."
                    if state == ReplayState.STOPPED else
                    "Recording loaded. Press Play to start.")
            self.view.show_placeholder(text)
            self.frame_label.setText(
                f"{len(session.frames)} frames · "
                f"{session.readiness_frame_count} readiness reports")
            self._show_report(None, unavailable_reason=text)
            return
        association = frame.association()
        image = None
        if frame.image_path is not None:
            try:
                image = self.cache.get(frame)
            except SessionError as exc:
                self._fail_session(str(exc))
                return
        report = frame.readiness
        draw = association.matched and image is not None
        if image is None:
            reason = ("RGB frame missing for this report."
                      if frame.image_path is None else
                      "The RGB frame could not be read.")
            self.view.show_placeholder(reason)
        else:
            notice = ""
            if report is not None and not association.matched:
                notice = ("Detections not drawn: " + association.reason)
            self.view.show_frame(
                image, report.records if report else [], draw, notice)
        label = f" · {frame.label}" if frame.label else ""
        self.frame_label.setText(
            f"Frame {frame.position + 1}/{len(session.frames)}{label} · "
            f"image stamp {format_stamp(frame.image_stamp_ns)}")
        if report is None:
            if session.readiness_frame_count == 0:
                reason = ("READINESS UNAVAILABLE\nThis recording has no "
                          "readiness topic. EPD publishes it only with "
                          "hybrid RGB-D refinement enabled.")
            else:
                reason = ("READINESS UNAVAILABLE\nNo readiness report was "
                          "recorded for this frame.")
            self._show_report(None, unavailable_reason=reason)
        else:
            self._show_report(report)

    def _render_replay_health(self, session, frame, state):
        health = replay_health(state, session, frame, self.load_error)
        if session is not None:
            health.messages = list(session.warnings) + health.messages
            if self.player.compressed_gaps:
                health.messages.append(
                    f"{self.player.compressed_gaps} long source gap(s) are "
                    "shortened during playback; stamps are unchanged.")
        self._apply_health(health)
        if session is None:
            source = "Recorded replay (nothing loaded)"
        elif session.is_synthetic:
            source = "Synthetic test fixture: not camera data"
        else:
            source = "Recorded replay: not a live camera"
        self.health.set_value("source", source)
        self.health.set_value(
            "session",
            "—" if session is None else
            f"{session.name} · {state.lower()} · "
            f"{len(session.frames)} frames")
        if session is None:
            readiness = "—"
        elif session.readiness_frame_count == 0:
            readiness = "Unavailable in this recording"
        else:
            readiness = (f"{session.readiness_frame_count} reports on "
                         f"{session.readiness_topic}")
        self.health.set_value("readiness", readiness)
        if frame is None:
            frames = "—"
        elif frame.image_path is None:
            frames = "RGB frame missing for this report"
        else:
            frames = (f"{frame.image_width} × {frame.image_height} · "
                      f"{frame.image_frame_id}")
        self.health.set_value("frames", frames)
        rate = session.report_rate_hz() if session is not None else None
        self.health.set_value(
            "rate", "Unavailable" if rate is None else
            f"{rate:.2f} Hz (recorded source stamps)")
        self.health.set_value(
            "latency", "Unavailable (not part of the readiness report)")

    # ------------------------------------------------------------------ live
    def _on_job_state(self, state, _message):
        # Deploy's legacy handler (connected first) resets the Run icon to
        # 100 px on every state change, which the fixed-size footer button
        # clips. Restore the refreshed footer's compact icon.
        self.deploy.run_button.setIconSize(QSize(24, 24))
        if state == JobState.RUNNING and self._running_since is None:
            self._running_since = self._clock()
        elif state != JobState.RUNNING:
            self._running_since = None
        if self.mode == self.MODE_LIVE:
            self._render_live(force=False)

    def _live_tick(self):
        snapshot = self.live_monitor.drain()
        self._live_error = snapshot["error"]
        self._live_publisher_count = snapshot["publisher_count"]
        if snapshot["arrivals"]:
            self._live_arrivals = snapshot["arrivals"]
        reports = snapshot["reports"]
        if reports:
            latest = reports[-1]
            self._live_report_at = latest.received_monotonic
            self._show_report(latest)
        self._render_live(force=False)

    def _render_live(self, force=False):
        if self.mode != self.MODE_LIVE:
            return
        state = self.deploy._job_controller.state
        now = self._clock()
        age = None
        if self._live_report_at is not None:
            age = max(0.0, now - self._live_report_at)
        running_for = 0.0
        if self._running_since is not None:
            running_for = now - self._running_since
        stale_limit = stale_after_s(self._live_arrivals)
        health = live_health(
            state.name if state != JobState.IDLE else "STOPPED",
            self._live_error, self._live_publisher_count, age,
            stale_limit, running_for)
        self._apply_health(health)
        if self.current_report is None or force:
            if health.state in ("UNAVAILABLE", "FAILED"):
                reason = "READINESS UNAVAILABLE\n" + " ".join(
                    [health.headline] + health.messages)
            elif state in (JobState.RUNNING, JobState.STARTING):
                reason = ("READINESS UNAVAILABLE\nNo readiness report "
                          "received yet. It is published only with hybrid "
                          "RGB-D refinement enabled.")
            else:
                reason = ("READINESS UNAVAILABLE\nPerception is stopped. "
                          "Run perception, or switch to Replay recording.")
            if self.current_report is None:
                self._show_report(None, unavailable_reason=reason)
        self.view.set_stale(False)
        self.health.set_value(
            "source", f"ROS topic {READINESS_TOPIC}")
        self.health.set_value(
            "session", f"Deploy {state.name.lower()}")
        if self._live_error:
            readiness = "Unavailable (ROS monitor error)"
        elif self._live_publisher_count:
            readiness = f"{self._live_publisher_count} publisher(s)"
        else:
            readiness = "No publisher (hybrid refinement off or EPD stopped)"
        self.health.set_value("readiness", readiness)
        self.health.set_value(
            "frames", "Shown in the live preview; not matched to reports "
            "by source stamp")
        arrivals = self._live_arrivals
        rate = None
        if len(arrivals) >= 2 and arrivals[-1] > arrivals[0]:
            rate = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
        self.health.set_value(
            "rate", "Unavailable" if rate is None else
            f"{rate:.2f} Hz (arrival times)")
        self.health.set_value(
            "latency", "Unavailable (not part of the readiness report)")
        if health.state == "STALE":
            self.summary_label.setText(
                "STALE · " + self._summary_text(self.current_report))
        self._update_ages()

    # ------------------------------------------------------------- reporting
    def _summary_text(self, report):
        if report is None:
            return ""
        counts = report.counts()
        parts = [f"{len(report.records)} proposals"]
        parts += [f"{counts[state]} {state}" for state in STATES
                  if counts[state]]
        return " · ".join(parts)

    def _show_report(self, report, unavailable_reason=""):
        previous_key = self.selected_key
        same_report = report is not None and report is self.current_report
        self.current_report = report
        self.current_records = list(report.records) if report else []
        if report is None:
            self._set_rows(None)
            self.unavailable_label.setText(unavailable_reason)
            self.table_stack.setCurrentIndex(1)
            self.summary_label.setText("")
            self._clear_selection()
            return
        self.table_stack.setCurrentIndex(0)
        self.summary_label.setText(
            self._summary_text(report) if report.records else
            "Report received with 0 detections")
        self._set_rows(report)
        keep_row = None
        for row, record in enumerate(report.records):
            if previous_key is not None and record.key == previous_key:
                keep_row = row
        # A detection index is not an identity across reports; only a track
        # ID is. Selection therefore follows tracks, or the same report.
        if keep_row is not None and (same_report or
                                     previous_key[0] == "track"):
            self._programmatic_selection = True
            try:
                self.table.selectRow(keep_row)
            finally:
                self._programmatic_selection = False
        else:
            self._clear_selection()

    def _set_rows(self, report):
        self._refreshing = True
        try:
            self.table_model.set_report(report)
        finally:
            self._refreshing = False

    def _update_ages(self):
        report = self.current_report
        if report is None:
            self.health.set_value("age", "—")
            self.health.set_value("objects", "—")
            return
        if self.mode == self.MODE_REPLAY:
            elapsed = self.player.frame_elapsed_s()
            age = ("—" if elapsed is None else f"{elapsed:.1f} s")
            age_text = ("—" if elapsed is None else
                        f"{elapsed:.1f} s on the replay clock")
        else:
            if self._live_report_at is None:
                age, age_text = "—", "—"
            else:
                value = max(0.0, self._clock() - self._live_report_at)
                age = f"{value:.1f} s"
                age_text = f"{value:.1f} s since the report arrived"
        self.table_model.set_age(age)
        self.health.set_value("age", age_text)
        counts = report.counts()
        objects = ", ".join(
            f"{counts[state]} {state}" for state in STATES if counts[state])
        self.health.set_value(
            "objects", f"{len(report.records)} proposals"
            + (f" ({objects})" if objects else ""))

    def _apply_health(self, health):
        self.health.set_health(health)
        _set_badge(self.health_badge, health.state)
        self.health_state = health

    # ------------------------------------------------------------- selection
    def _association_text(self):
        if self.mode == self.MODE_LIVE:
            return ("Not shown: the live preview frame is not matched to "
                    "this report by source stamp.")
        frame = self.current_frame
        if frame is None:
            return UNAVAILABLE
        association = frame.association()
        if not association.matched:
            return "Not matched. " + association.reason
        record = self._selected_record()
        if record is not None and not roi_inside(
                record.roi, frame.image_width, frame.image_height):
            return ("Matched, but the ROI lies outside the image, so it is "
                    "not drawn.")
        return "Matched. " + association.reason

    def _selected_record(self):
        for record in self.current_records:
            if record.key == self.selected_key:
                return record
        return None

    def _on_table_selection(self, *_args):
        if self._refreshing:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows or self.current_report is None:
            self.selected_key = None
            self.view.set_selected(None)
            self.inspector.clear()
            return
        record = self.current_records[rows[0].row()]
        self.selected_key = record.key
        self.view.set_selected(record.key)
        self.inspector.show_record(
            record, self.current_report, self._association_text())
        if (not self._programmatic_selection and
                self.side_tabs.currentWidget() is not self.inspector):
            self.side_tabs.setCurrentWidget(self.inspector)

    def select_row(self, row):
        self.table.selectRow(row)

    def _clear_selection(self):
        self.selected_key = None
        if hasattr(self, "table"):
            self._refreshing = True
            try:
                self.table.clearSelection()
            finally:
                self._refreshing = False
        if hasattr(self, "view"):
            self.view.set_selected(None)
        if hasattr(self, "inspector"):
            self.inspector.clear()

    # --------------------------------------------------------------- general
    def _update_controls(self):
        state = self.player.state
        loaded = self.session is not None
        self.play_button.setEnabled(
            loaded and state != ReplayState.PLAYING)
        self.pause_button.setEnabled(state == ReplayState.PLAYING)
        self.restart_button.setEnabled(loaded)
        self.stop_button.setEnabled(
            loaded and state not in (ReplayState.STOPPED, ReplayState.READY))
        stepping = loaded and state in (
            ReplayState.PAUSED, ReplayState.READY, ReplayState.ENDED)
        self.prev_button.setEnabled(stepping)
        self.next_button.setEnabled(stepping)
        if self.mode == self.MODE_LIVE:
            self.position_label.setText(
                "Live: camera preview and the EPD readiness topic")
        elif not loaded:
            self.position_label.setText("No recording loaded")
        else:
            index = self.player.index
            total = len(self.session.frames)
            shown = "—" if index is None else str(index + 1)
            stamp = ("" if index is None else " · source " + format_stamp(
                self.session.frames[index].stamp_ns))
            self.position_label.setText(
                f"{state.title()} · frame {shown}/{total} · "
                f"{self.player.elapsed_s():.1f}/"
                f"{self.player.duration_s:.1f} s{stamp}")

    def eventFilter(self, obj, event):
        if obj is self.deploy and event.type() == QEvent.Show:
            if self.mode == self.MODE_LIVE:
                QTimer.singleShot(0, self._start_live_monitor)
            else:
                QTimer.singleShot(0, self._stop_live_preview)
        elif obj is self.deploy and event.type() in (QEvent.Hide,
                                                     QEvent.Close):
            self._release_runtime()
        return super().eventFilter(obj, event)

    def _release_runtime(self):
        if self.player.state == ReplayState.PLAYING:
            self.pause()
        self._stop_live_monitor()
        self.live_monitor.stop()
        self.cache.clear()

    def workers_alive(self):
        return self.live_monitor.is_running()

    def live_subscribed(self):
        return self.live_monitor.subscribed()

    def shutdown(self):
        if self._is_shut_down:
            return
        self._is_shut_down = True
        self.replay_timer.stop()
        self.live_timer.stop()
        self.player.stop()
        self.live_monitor.stop()
        self.cache.clear()

    def _append_style(self):
        self.deploy.setStyleSheet(self.deploy.styleSheet() + """
            QWidget#perceptionWorkspace { background-color: #0d1117; }
            QFrame#monitorTransport {
                background-color: #121821;
                border: 1px solid #263140;
                border-radius: 10px;
            }
            QPushButton#monitorModeButton, QPushButton#monitorTransportButton {
                min-height: 18px;
                padding: 5px 10px;
            }
            QToolButton#monitorOpenButton {
                color: #dce3ed;
                background-color: #121720;
                border: 1px solid #303a49;
                border-radius: 7px;
                padding: 5px 22px 5px 10px;
                font-size: 11px;
                min-height: 18px;
            }
            QToolButton#monitorOpenButton:hover {
                background-color: #1c2330;
                border-color: #536278;
            }
            QToolButton#monitorOpenButton::menu-button {
                border-left: 1px solid #303a49;
                width: 16px;
            }
            QTabBar::tab {
                color: #9aa7b8;
                background-color: #121720;
                border: 1px solid #263140;
                border-bottom: none;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                padding: 6px 14px;
                font-size: 11px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                color: #e4eaf2;
                background-color: #1a2230;
                border-color: #3a4a62;
            }
            QHeaderView::section {
                color: #8c9ab0;
                background-color: #141a23;
                border: none;
                border-right: 1px solid #222b37;
                border-bottom: 1px solid #263140;
                padding: 4px 6px;
                font-size: 10px;
                font-weight: 700;
            }
            QPushButton#monitorModeButton { font-weight: 600; }
            QPushButton#monitorModeButton:checked {
                color: #0b0f14;
                background-color: #8ab4f8;
                border-color: #8ab4f8;
            }
            QLabel#monitorPosition { color: #9aa7b8; font-size: 10px; }
            QLabel#monitorHealthBadge {
                color: #aeb8c7;
                background-color: #141a23;
                border: 1px solid #313c4b;
                border-radius: 7px;
                padding: 5px 9px;
                font-size: 9px;
                font-weight: 700;
            }
            QLabel#monitorHealthBadge[monitorState="live"] {
                color: #9de2b2; background-color: #14241b;
                border-color: #285b39;
            }
            QLabel#monitorHealthBadge[monitorState="replay"] {
                color: #a9c7fb; background-color: #142034;
                border-color: #2d4c7c;
            }
            QLabel#monitorHealthBadge[monitorState="waiting"] {
                color: #e8c98f; background-color: #292116;
                border-color: #634a24;
            }
            QLabel#monitorHealthBadge[monitorState="failed"] {
                color: #e3a1a1; background-color: #2a1719;
                border-color: #653238;
            }
            QLabel#monitorProvenance {
                color: #aeb8c7;
                background-color: #141a23;
                border: 1px solid #313c4b;
                border-radius: 6px;
                padding: 3px 8px;
                font-size: 9px;
                font-weight: 700;
            }
            QLabel#monitorProvenance[provenance="recorded"] {
                color: #a9c7fb; background-color: #142034;
                border-color: #2d4c7c;
            }
            QLabel#monitorProvenance[provenance="synthetic"] {
                color: #d7b4f3; background-color: #241a30;
                border-color: #5b3f7a;
            }
            QTableView#readinessTable {
                background-color: #0f141b;
                alternate-background-color: #121821;
                gridline-color: #222b37;
                color: #dfe6ef;
                font-size: 11px;
                selection-background-color: #24344d;
            }
            QLabel#readinessUnavailable {
                color: #e8c98f;
                background-color: #17140f;
                border: 1px dashed #634a24;
                border-radius: 8px;
                padding: 12px;
                font-size: 11px;
                font-weight: 600;
            }
            QWidget#monitorPanelBody { background-color: #0f141b; }
            QLabel#inspectorTitle {
                color: #e4eaf2; font-size: 14px; font-weight: 700;
            }
            QLabel#monitorField { color: #7d8aa0; font-size: 10px; }
            QLabel#inspectorUsable {
                color: #c9d4e3;
                background-color: #151c27;
                border: 1px solid #2b3647;
                border-radius: 6px;
                padding: 5px 8px;
                font-size: 11px;
                font-weight: 600;
            }
            QLabel#monitorValue { color: #dfe6ef; font-size: 11px; }
            QLabel#monitorHint, QLabel#monitorHeadline {
                color: #9aa7b8; font-size: 11px;
            }
            QLabel#monitorMessages { color: #9aa7b8; font-size: 11px; }
            QLabel#monitorMessages[state="warn"] { color: #e8c98f; }
            QTabWidget#monitorSideTabs::pane {
                border: 1px solid #263140; border-radius: 8px;
            }
            """)


def apply_perception_monitor(main_window):
    """Install the Perception Monitor workspace in Deploy (idempotent)."""
    if getattr(main_window, "_perception_monitor", None) is not None:
        return main_window._perception_monitor
    controller = PerceptionMonitorController(main_window)
    main_window._perception_monitor = controller
    help_window = getattr(main_window, "help_window", None)
    if help_window is not None and "Perception Monitor" not in getattr(
            help_window, "TOPICS", {}):
        help_window.TOPICS["Perception Monitor"] = _HELP_TOPIC
        help_window.topic_list.addItem("Perception Monitor")
    return controller


_HELP_TOPIC = (
    "<h2>Perception Monitor</h2>"
    "<p>Deploy shows what EPD sees and why each detection is or is not "
    "usable in 3D.</p>"
    "<h3>Replay without a camera</h3>"
    "<p>Choose <b>Replay recording</b>, then <b>Open recording…</b>. A "
    "recording is a folder with <code>session.json</code>, exported from a "
    "rosbag2 with <code>epd_perception_session_export.py</code>. Bundled "
    "<i>synthetic test fixtures</i> are drawn shapes, not camera data.</p>"
    "<h3>Readiness states</h3>"
    "<ul><li><b>REJECTED</b>: 2D evidence unusable or contested.</li>"
    "<li><b>DETECTED</b>: class and mask, no 3D position.</li>"
    "<li><b>LOCALISED</b>: 3D position, orientation not valid.</li>"
    "<li><b>POSE_VALID</b>: full pose, not published (see note).</li>"
    "<li><b>GRASP_CANDIDATE</b>: published for downstream planning. "
    "This does not mean the grasp is reachable, collision-free or will "
    "succeed.</li></ul>"
    "<p>Boxes are drawn only when the image and the report share the same "
    "source stamp. The readiness topic is published only with hybrid "
    "RGB-D refinement enabled; otherwise Deploy shows READINESS "
    "UNAVAILABLE.</p>"
)
