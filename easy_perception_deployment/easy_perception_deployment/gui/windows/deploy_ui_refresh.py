from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSettings, QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class _DeployUiController(QObject):
    """Presentation-only controller for the existing DeployWindow widgets."""

    SETTINGS_ORG = "WorkcellStudio"
    SETTINGS_APP = "EasyPerceptionDeployment"
    SETTINGS_KEY = "deploy_window/size"

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        self._summary_timer = QTimer(self)
        self._summary_timer.setInterval(300)
        self._summary_timer.timeout.connect(self.sync)

        self.model_value = None
        self.labels_value = None
        self.topic_value = None
        self.model_state = None
        self.labels_state = None
        self.topic_state = None
        self.header_badge = None

    def apply(self):
        self._configure_window()
        self._clear_legacy_layout()
        self._configure_existing_widgets()
        self._build_layout()
        self._apply_style()
        self._connect_refresh_hooks()
        self.window.installEventFilter(self)
        self.sync()
        self._summary_timer.start()

    def _configure_window(self):
        self.window.setObjectName("deployWindow")
        self.window.setWindowTitle("Deploy Perception")

        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen else None

        default = QSize(900, 700)
        saved = self.settings.value(self.SETTINGS_KEY)
        if isinstance(saved, QSize) and saved.isValid():
            default = saved

        if available is not None:
            max_width = max(680, int(available.width() * 0.94))
            max_height = max(540, int(available.height() * 0.92))
            width = min(max(default.width(), 720), max_width)
            height = min(max(default.height(), 580), max_height)
            min_width = min(720, max_width)
            min_height = min(560, max_height)
        else:
            width, height = default.width(), default.height()
            min_width, min_height = 720, 560

        self.window.setMinimumSize(min_width, min_height)
        self.window.resize(width, height)

    def _clear_legacy_layout(self):
        root = self.window.layout()
        if root is None:
            root = QGridLayout(self.window)
        while root.count():
            root.takeAt(0)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.root_layout = root

    def _configure_existing_widgets(self):
        w = self.window

        # Remove legacy red/green full-surface signalling. Readiness is shown
        # separately as compact status chips in this refreshed layout.
        w.model_button.setStyleSheet("")
        w.list_button.setStyleSheet("")
        w.usecase_config_button.setStyleSheet("")

        w.model_button.setText("Change model")
        w.list_button.setText("Change labels")
        w.model_button.setIconSize(QSize(26, 26))
        w.list_button.setIconSize(QSize(26, 26))
        w.model_button.setMinimumHeight(40)
        w.list_button.setMinimumHeight(40)

        w.visualize_button.setMinimumHeight(40)
        w.segmentation_button.setMinimumHeight(40)
        w.refresh_topics_button.setMinimumHeight(38)
        w.use_defaults_button.setMinimumHeight(36)
        w.docker_button.setMinimumHeight(40)

        w.topic_button.setMinimumHeight(40)
        w.topic_button.setFixedHeight(40)
        w.transport_combo.setMinimumHeight(38)
        w.usecase_config_button.setMinimumHeight(40)
        w.confidence_spinbox.setMinimumHeight(38)
        w.max_detections_spinbox.setMinimumHeight(38)

        w.run_button.setMinimumHeight(50)
        w.run_button.setIconSize(QSize(24, 24))
        w.run_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        w.validation_label.setWordWrap(True)
        w.validation_label.setObjectName("validationMessage")
        w.status_label.setIndent(0)
        w.status_label.setObjectName("runtimeStatus")
        w.fps_label.setIndent(0)
        w.fps_label.setObjectName("runtimeMetrics")

        # These still receive updates from DeployWindow and remain the truth
        # source, but their verbose absolute paths are replaced by compact rows.
        w.readiness_header_label.hide()
        w.model_readiness_label.hide()
        w.label_list_readiness_label.hide()
        w.topic_readiness_label.hide()

    def _build_layout(self):
        w = self.window

        header = QFrame(w)
        header.setObjectName("deployHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 18, 24, 16)
        header_layout.setSpacing(14)

        title_stack = QVBoxLayout()
        title_stack.setContentsMargins(0, 0, 0, 0)
        title_stack.setSpacing(2)

        eyebrow = QLabel("EPD  •  LIVE PERCEPTION", header)
        eyebrow.setObjectName("deployEyebrow")
        title = QLabel("Deploy Perception", header)
        title.setObjectName("deployTitle")
        subtitle = QLabel(
            "Configure the model, camera input and perception mode, then run safely.",
            header,
        )
        subtitle.setObjectName("deploySubtitle")
        subtitle.setWordWrap(True)

        title_stack.addWidget(eyebrow)
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header_layout.addLayout(title_stack, 1)

        self.header_badge = QLabel("CHECKING", header)
        self.header_badge.setObjectName("headerBadge")
        self.header_badge.setAlignment(Qt.AlignCenter)
        self.header_badge.setMinimumWidth(116)
        self.header_badge.setMinimumHeight(30)
        header_layout.addWidget(self.header_badge, 0, Qt.AlignTop)

        self.root_layout.addWidget(header, 0, 0)

        scroll = QScrollArea(w)
        scroll.setObjectName("deployScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget(scroll)
        content.setObjectName("deployContent")
        content_grid = QGridLayout(content)
        content_grid.setContentsMargins(22, 18, 22, 18)
        content_grid.setHorizontalSpacing(14)
        content_grid.setVerticalSpacing(14)
        content_grid.setColumnStretch(0, 1)
        content_grid.setColumnStretch(1, 1)

        model_card, model_layout = self._section(
            content,
            "MODEL",
            "Choose the ONNX model and label list used by the pipeline.",
        )
        self.model_value, self.model_state = self._summary_row(model_card)
        self.labels_value, self.labels_state = self._summary_row(model_card)
        model_layout.addLayout(self._row_with_label("Model", self.model_value, self.model_state))
        model_layout.addLayout(self._row_with_label("Labels", self.labels_value, self.labels_state))

        model_actions = QHBoxLayout()
        model_actions.setSpacing(8)
        model_actions.addWidget(w.model_button)
        model_actions.addWidget(w.list_button)
        model_layout.addLayout(model_actions)

        perception_card, perception_layout = self._section(
            content,
            "PERCEPTION",
            "Select the use case and runtime output options.",
        )
        perception_layout.addLayout(
            self._form_row("Mode", w.usecase_config_button)
        )
        toggles = QHBoxLayout()
        toggles.setSpacing(8)
        toggles.addWidget(w.visualize_button)
        toggles.addWidget(w.segmentation_button)
        perception_layout.addLayout(toggles)

        camera_card, camera_layout = self._section(
            content,
            "CAMERA INPUT",
            "Select an image topic from ROS 2 or type one manually.",
        )
        topic_row = QHBoxLayout()
        topic_row.setSpacing(8)
        topic_row.addWidget(w.topic_button, 1)
        topic_row.addWidget(w.refresh_topics_button)
        camera_layout.addLayout(topic_row)

        transport_row = QHBoxLayout()
        transport_row.setSpacing(10)
        transport_row.addWidget(w.transport_label)
        transport_row.addStretch(1)
        transport_row.addWidget(w.transport_combo)
        camera_layout.addLayout(transport_row)

        runtime_card, runtime_layout = self._section(
            content,
            "RUNTIME",
            "Execution device and detection limits.",
        )
        runtime_layout.addWidget(w.docker_button)
        runtime_layout.addLayout(
            self._form_row(w.confidence_label.text(), w.confidence_spinbox)
        )
        runtime_layout.addLayout(
            self._form_row(w.max_detections_label.text(), w.max_detections_spinbox)
        )
        runtime_layout.addWidget(w.use_defaults_button, 0, Qt.AlignLeft)

        readiness_card, readiness_layout = self._section(
            content,
            "READINESS",
            "Run stays blocked until required inputs are valid.",
        )
        readiness_layout.addWidget(w.validation_label)

        self.topic_value, self.topic_state = self._summary_row(readiness_card)
        readiness_layout.addLayout(
            self._row_with_label("Model", self._duplicate_value("model"), self._duplicate_state("model"))
        )
        readiness_layout.addLayout(
            self._row_with_label("Labels", self._duplicate_value("labels"), self._duplicate_state("labels"))
        )
        readiness_layout.addLayout(
            self._row_with_label("Input", self.topic_value, self.topic_state)
        )

        content_grid.addWidget(model_card, 0, 0)
        content_grid.addWidget(perception_card, 0, 1)
        content_grid.addWidget(camera_card, 1, 0, 1, 2)
        content_grid.addWidget(runtime_card, 2, 0)
        content_grid.addWidget(readiness_card, 2, 1)

        scroll.setWidget(content)
        self.root_layout.addWidget(scroll, 1, 0)
        self.root_layout.setRowStretch(1, 1)

        footer = QFrame(w)
        footer.setObjectName("deployFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(22, 12, 22, 12)
        footer_layout.setSpacing(14)

        status_stack = QVBoxLayout()
        status_stack.setContentsMargins(0, 0, 0, 0)
        status_stack.setSpacing(1)
        status_stack.addWidget(w.status_label)
        status_stack.addWidget(w.fps_label)

        footer_layout.addLayout(status_stack, 1)
        footer_layout.addWidget(w.run_button, 0, Qt.AlignVCenter)
        self.root_layout.addWidget(footer, 2, 0)

        # Store duplicate readiness row widgets so they can mirror the model
        # card without exposing absolute file-system paths.
        self.readiness_model_value = self._duplicates["model_value"]
        self.readiness_model_state = self._duplicates["model_state"]
        self.readiness_labels_value = self._duplicates["labels_value"]
        self.readiness_labels_state = self._duplicates["labels_state"]

    def _section(self, parent, title_text, subtitle_text):
        frame = QFrame(parent)
        frame.setObjectName("sectionCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title = QLabel(title_text, frame)
        title.setObjectName("sectionTitle")
        subtitle = QLabel(subtitle_text, frame)
        subtitle.setObjectName("sectionSubtitle")
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        return frame, layout

    def _summary_row(self, parent):
        value = QLabel(parent)
        value.setObjectName("summaryValue")
        value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        state = QLabel("•", parent)
        state.setObjectName("summaryState")
        state.setAlignment(Qt.AlignCenter)
        state.setMinimumWidth(74)
        return value, state

    def _row_with_label(self, label_text, value_widget, state_widget):
        row = QHBoxLayout()
        row.setSpacing(10)
        label = QLabel(label_text)
        label.setObjectName("fieldLabel")
        label.setMinimumWidth(48)
        row.addWidget(label)
        row.addWidget(value_widget, 1)
        row.addWidget(state_widget)
        return row

    def _form_row(self, label_text, widget):
        row = QHBoxLayout()
        row.setSpacing(10)
        label = QLabel(label_text)
        label.setObjectName("fieldLabel")
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(widget)
        return row

    def _duplicate_value(self, key):
        if not hasattr(self, "_duplicates"):
            self._duplicates = {}
        label = QLabel()
        label.setObjectName("summaryValue")
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._duplicates[f"{key}_value"] = label
        return label

    def _duplicate_state(self, key):
        if not hasattr(self, "_duplicates"):
            self._duplicates = {}
        label = QLabel("•")
        label.setObjectName("summaryState")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumWidth(74)
        self._duplicates[f"{key}_state"] = label
        return label

    def _connect_refresh_hooks(self):
        w = self.window
        for signal_owner in (
            w.model_button,
            w.list_button,
            w.visualize_button,
            w.segmentation_button,
            w.docker_button,
            w.refresh_topics_button,
            w.use_defaults_button,
        ):
            signal_owner.clicked.connect(self._sync_soon)
        w.topic_button.currentTextChanged.connect(self._sync_soon)
        w.usecase_config_button.currentTextChanged.connect(self._sync_soon)

    def _sync_soon(self, *args):
        QTimer.singleShot(0, self.sync)
        QTimer.singleShot(450, self.sync)

    def sync(self):
        w = self.window

        model_name = Path(str(getattr(w, "_path_to_model", ""))).name or "Not configured"
        labels_name = Path(str(getattr(w, "_path_to_label_list", ""))).name or "Not configured"
        topic = w.topic_button.currentText().strip() or "Not configured"

        model_status = self._status_from_text(w.model_readiness_label.text())
        labels_status = self._status_from_text(w.label_list_readiness_label.text())
        topic_status = self._status_from_text(w.topic_readiness_label.text())

        self.model_value.setText(model_name)
        self.model_value.setToolTip(str(getattr(w, "_path_to_model", "")))
        self.labels_value.setText(labels_name)
        self.labels_value.setToolTip(str(getattr(w, "_path_to_label_list", "")))
        self.topic_value.setText(topic)
        self.topic_value.setToolTip(topic)

        self.readiness_model_value.setText(model_name)
        self.readiness_model_value.setToolTip(str(getattr(w, "_path_to_model", "")))
        self.readiness_labels_value.setText(labels_name)
        self.readiness_labels_value.setToolTip(str(getattr(w, "_path_to_label_list", "")))

        self._set_state(self.model_state, model_status)
        self._set_state(self.labels_state, labels_status)
        self._set_state(self.topic_state, topic_status)
        self._set_state(self.readiness_model_state, model_status)
        self._set_state(self.readiness_labels_state, labels_status)

        w.visualize_button.setText(
            "Visual output  •  On" if getattr(w, "visualizeFlag", False)
            else "Visual output  •  Off"
        )
        w.segmentation_button.setText(
            "Segmentation  •  On"
            if getattr(w, "publish_detection_segmentation", False)
            else "Segmentation  •  Off"
        )
        w.docker_button.setText(
            "Device  •  CPU" if getattr(w, "useCPU", True)
            else "Device  •  GPU"
        )

        is_running = bool(getattr(w, "_is_running", False))
        if is_running:
            self.header_badge.setText("RUNNING")
            self.header_badge.setProperty("state", "running")
            w.run_button.setText("Stop Perception")
        elif w.run_button.isEnabled():
            self.header_badge.setText("READY")
            self.header_badge.setProperty("state", "ready")
            w.run_button.setText("Run Perception")
        else:
            self.header_badge.setText("SETUP REQUIRED")
            self.header_badge.setProperty("state", "blocked")
            w.run_button.setText("Run Perception")

        self.header_badge.style().unpolish(self.header_badge)
        self.header_badge.style().polish(self.header_badge)

    def _status_from_text(self, text):
        text = text or ""
        if "✅" in text or "ready" in text.lower() or "valid" in text.lower():
            return "ready"
        if "❌" in text or "⚠" in text or "missing" in text.lower() or "invalid" in text.lower():
            return "blocked"
        return "unknown"

    def _set_state(self, label, state):
        if state == "ready":
            label.setText("✓ Ready")
            label.setProperty("state", "ready")
        elif state == "blocked":
            label.setText("! Check")
            label.setProperty("state", "blocked")
        else:
            label.setText("• Checking")
            label.setProperty("state", "unknown")
        label.style().unpolish(label)
        label.style().polish(label)

    def _apply_style(self):
        self.window.setStyleSheet(
            """
            QWidget#deployWindow,
            QWidget#deployContent {
                background-color: #101319;
                color: #e8edf5;
            }

            QFrame#deployHeader {
                background-color: #121720;
                border-bottom: 1px solid #27303d;
            }

            QLabel#deployEyebrow {
                color: #7e8ca3;
                font-size: 10px;
                font-weight: 700;
            }

            QLabel#deployTitle {
                color: #f7f9fc;
                font-size: 24px;
                font-weight: 700;
            }

            QLabel#deploySubtitle {
                color: #a9b4c4;
                font-size: 12px;
            }

            QLabel#headerBadge {
                color: #aab5c5;
                background-color: #1a202a;
                border: 1px solid #313b4a;
                border-radius: 8px;
                padding: 5px 10px;
                font-size: 10px;
                font-weight: 700;
            }

            QLabel#headerBadge[state="ready"] {
                color: #9de2b2;
                background-color: #14241b;
                border-color: #285b39;
            }

            QLabel#headerBadge[state="running"] {
                color: #b8c8ff;
                background-color: #17203a;
                border-color: #40599c;
            }

            QLabel#headerBadge[state="blocked"] {
                color: #e8c98f;
                background-color: #292116;
                border-color: #634a24;
            }

            QScrollArea#deployScroll {
                background-color: #101319;
                border: none;
            }

            QScrollArea#deployScroll > QWidget > QWidget {
                background-color: #101319;
            }

            QFrame#sectionCard {
                background-color: #181d26;
                border: 1px solid #2b3441;
                border-radius: 12px;
            }

            QLabel#sectionTitle {
                color: #8c9ab0;
                font-size: 10px;
                font-weight: 700;
            }

            QLabel#sectionSubtitle {
                color: #758196;
                font-size: 11px;
            }

            QLabel#fieldLabel {
                color: #9ba7b8;
                font-size: 12px;
            }

            QLabel#summaryValue {
                color: #e8edf5;
                font-size: 12px;
            }

            QLabel#summaryState {
                color: #8793a5;
                background-color: #121720;
                border: 1px solid #2d3644;
                border-radius: 7px;
                padding: 4px 7px;
                font-size: 10px;
                font-weight: 600;
            }

            QLabel#summaryState[state="ready"] {
                color: #98dfad;
                background-color: #14231a;
                border-color: #2b5b3a;
            }

            QLabel#summaryState[state="blocked"] {
                color: #e4c28a;
                background-color: #261f15;
                border-color: #5c4626;
            }

            QLabel#validationMessage {
                color: #b5c0cf;
                background-color: #121720;
                border: 1px solid #293240;
                border-radius: 8px;
                padding: 8px 10px;
                font-size: 11px;
            }

            QPushButton,
            QComboBox,
            QDoubleSpinBox,
            QSpinBox {
                color: #dce3ed;
                background-color: #121720;
                border: 1px solid #303a49;
                border-radius: 7px;
                padding: 6px 9px;
                font-size: 12px;
            }

            QPushButton:hover,
            QComboBox:hover,
            QDoubleSpinBox:hover,
            QSpinBox:hover {
                background-color: #1c2330;
                border-color: #536278;
            }

            QPushButton:pressed {
                background-color: #151b24;
            }

            QPushButton:disabled,
            QComboBox:disabled,
            QDoubleSpinBox:disabled,
            QSpinBox:disabled {
                color: #626e80;
                background-color: #141820;
                border-color: #252c37;
            }

            QComboBox QAbstractItemView {
                color: #e3e8ef;
                background-color: #171c24;
                border: 1px solid #354052;
                selection-background-color: #28354b;
            }

            QFrame#deployFooter {
                background-color: #121720;
                border-top: 1px solid #27303d;
            }

            QLabel#runtimeStatus {
                color: #d4dbe6;
                font-size: 12px;
                font-weight: 600;
            }

            QLabel#runtimeMetrics {
                color: #7f8b9e;
                font-size: 11px;
            }

            QPushButton#runButton {
                color: #ffffff;
                background-color: #5367d8;
                border: 1px solid #6e80e6;
                border-radius: 9px;
                padding: 9px 18px;
                font-size: 13px;
                font-weight: 700;
            }

            QPushButton#runButton:hover {
                background-color: #6275e2;
                border-color: #8796ee;
            }
            """
        )
        self.window.run_button.setObjectName("runButton")
        self.window.run_button.style().unpolish(self.window.run_button)
        self.window.run_button.style().polish(self.window.run_button)

    def eventFilter(self, obj, event):
        if obj is self.window and event.type() in (QEvent.Hide, QEvent.Close):
            if not self.window.isMaximized():
                self.settings.setValue(self.SETTINGS_KEY, self.window.size())
        return super().eventFilter(obj, event)


def apply_deploy_ui_refresh(window):
    """Apply a presentation-only refresh to an existing DeployWindow."""
    controller = _DeployUiController(window)
    controller.apply()
    window._deploy_ui_controller = controller
    return controller
