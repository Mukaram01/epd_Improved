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


class _TrainUiController(QObject):
    """Presentation-only controller for the existing TrainWindow widgets."""

    SETTINGS_ORG = "WorkcellStudio"
    SETTINGS_APP = "EasyPerceptionDeployment"
    SETTINGS_KEY = "train_window/size"

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        self._summary_timer = QTimer(self)
        self._summary_timer.setInterval(300)
        self._summary_timer.timeout.connect(self.sync)

        self.header_badge = None
        self.readiness_message = None
        self.model_value = None
        self.dataset_value = None
        self.labels_value = None
        self.model_state = None
        self.dataset_state = None
        self.labels_state = None
        self.annotations_state = None

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
        self.window.setObjectName("trainWindow")
        self.window.setWindowTitle("Train Vision Model")

        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen else None

        default = QSize(960, 740)
        saved = self.settings.value(self.SETTINGS_KEY)
        if isinstance(saved, QSize) and saved.isValid():
            default = saved

        if available is not None:
            max_width = max(760, int(available.width() * 0.94))
            max_height = max(600, int(available.height() * 0.92))
            width = min(max(default.width(), 780), max_width)
            height = min(max(default.height(), 620), max_height)
            min_width = min(760, max_width)
            min_height = min(580, max_height)
        else:
            width, height = default.width(), default.height()
            min_width, min_height = 760, 580

        self.window.setMinimumSize(min_width, min_height)
        self.window.resize(width, height)

    def _clear_legacy_layout(self):
        root = self.window.layout()
        if root is None:
            root = QVBoxLayout(self.window)
        while root.count():
            root.takeAt(0)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.root_layout = root

    def _configure_existing_widgets(self):
        w = self.window

        # The legacy UI encodes state by painting entire controls red/green.
        # The refreshed UI moves state into compact readiness chips instead.
        self._clear_legacy_inline_styles()

        w.training_config_label.hide()

        for button in (
            w.p2_button,
            w.p3_button,
            w.model_selector,
            w.list_button,
            w.dataset_button,
            w.label_button,
            w.generate_button,
            w.validate_button,
            w.maxiter_button,
            w.checkpointp_button,
            w.steps_button,
            w.testp_button,
        ):
            button.setMinimumHeight(40)

        w.p2_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        w.p3_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        w.p2_button.setMinimumWidth(76)
        w.p3_button.setMinimumWidth(76)

        w.model_selector.setMinimumHeight(42)
        w.model_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        w.list_button.setIconSize(QSize(26, 26))
        w.dataset_button.setIconSize(QSize(26, 26))
        w.label_button.setIconSize(QSize(26, 26))
        w.generate_button.setIconSize(QSize(26, 26))
        w.validate_button.setIconSize(QSize(26, 26))

        w.train_button.setObjectName("trainPrimary")
        w.train_button.setMinimumHeight(52)
        w.train_button.setIconSize(QSize(26, 26))
        w.train_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        w.validate_button.setObjectName("validateAction")
        w.training_status_label.setObjectName("trainRuntimeStatus")
        w.training_status_label.setWordWrap(True)
        w.training_status_label.setMinimumHeight(0)

    def _build_layout(self):
        w = self.window

        header = QFrame(w)
        header.setObjectName("trainHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 18, 24, 16)
        header_layout.setSpacing(14)

        title_stack = QVBoxLayout()
        title_stack.setContentsMargins(0, 0, 0, 0)
        title_stack.setSpacing(2)

        eyebrow = QLabel("EPD  •  MODEL TRAINING", header)
        eyebrow.setObjectName("trainEyebrow")
        title = QLabel("Train Vision Model", header)
        title.setObjectName("trainTitle")
        subtitle = QLabel(
            "Choose a model, prepare the dataset, validate readiness and start training.",
            header,
        )
        subtitle.setObjectName("trainSubtitle")
        subtitle.setWordWrap(True)

        title_stack.addWidget(eyebrow)
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header_layout.addLayout(title_stack, 1)

        self.header_badge = QLabel("SETUP REQUIRED", header)
        self.header_badge.setObjectName("trainHeaderBadge")
        self.header_badge.setAlignment(Qt.AlignCenter)
        self.header_badge.setMinimumWidth(126)
        self.header_badge.setMinimumHeight(30)
        header_layout.addWidget(self.header_badge, 0, Qt.AlignTop)

        self.root_layout.addWidget(header)

        scroll = QScrollArea(w)
        scroll.setObjectName("trainScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget(scroll)
        content.setObjectName("trainContent")
        content_grid = QGridLayout(content)
        content_grid.setContentsMargins(22, 18, 22, 18)
        content_grid.setHorizontalSpacing(14)
        content_grid.setVerticalSpacing(14)
        content_grid.setColumnStretch(0, 1)
        content_grid.setColumnStretch(1, 1)

        model_card, model_layout = self._section(
            content,
            "MODEL",
            "Choose the training precision level and pretrained architecture.",
        )

        precision_row = QHBoxLayout()
        precision_row.setSpacing(8)
        precision_label = QLabel("Precision", model_card)
        precision_label.setObjectName("trainFieldLabel")
        precision_row.addWidget(precision_label)
        precision_row.addStretch(1)
        w.p2_button.setObjectName("precisionButton")
        w.p3_button.setObjectName("precisionButton")

        w.p2_button.setToolTip(
            "FP32 precision: higher numerical precision with increased memory usage."
        )
        w.p3_button.setToolTip(
            "FP16 precision: reduced memory usage and faster training on supported hardware."
        )
        precision_row.addWidget(w.p2_button)
        precision_row.addWidget(w.p3_button)
        model_layout.addLayout(precision_row)

        model_select_row = QHBoxLayout()
        model_select_row.setSpacing(10)
        model_select_label = QLabel("Architecture", model_card)
        model_select_label.setObjectName("trainFieldLabel")
        model_select_row.addWidget(model_select_label)
        model_select_row.addWidget(w.model_selector, 1)
        model_layout.addLayout(model_select_row)

        self.model_value, self.model_state = self._summary_row(model_card)
        model_layout.addLayout(
            self._row_with_label("Selected", self.model_value, self.model_state)
        )

        dataset_card, dataset_layout = self._section(
            content,
            "DATASET",
            "Link a dataset and labels, then prepare annotations for training.",
        )

        self.dataset_value, self.dataset_state = self._summary_row(dataset_card)
        self.labels_value, self.labels_state = self._summary_row(dataset_card)
        dataset_layout.addLayout(
            self._row_with_label("Dataset", self.dataset_value, self.dataset_state)
        )
        dataset_layout.addLayout(
            self._row_with_label("Labels", self.labels_value, self.labels_state)
        )

        choose_row = QHBoxLayout()
        choose_row.setSpacing(8)
        w.dataset_button.setText("Choose dataset")
        w.list_button.setText("Choose label list")

        w.dataset_button.setToolTip(
            "Select the labelled dataset containing training images and annotations."
        )
        w.list_button.setToolTip(
            "Select the label list matching the dataset classes used during training."
        )
        choose_row.addWidget(w.dataset_button)
        choose_row.addWidget(w.list_button)
        dataset_layout.addLayout(choose_row)

        prep_row = QHBoxLayout()
        prep_row.setSpacing(8)
        w.generate_button.setText("Generate COCO dataset")
        w.label_button.setToolTip(
            "Select or prepare label information used by the training pipeline."
        )
        w.generate_button.setToolTip(
            "Convert labelled data into the COCO format required by training."
        )
        prep_row.addWidget(w.label_button)
        prep_row.addWidget(w.generate_button)
        dataset_layout.addLayout(prep_row)

        params_card, params_layout = self._section(
            content,
            "TRAINING PARAMETERS",
            "Adjust the main trainer values. Click any value to edit it.",
        )
        params_grid = QGridLayout()
        params_grid.setHorizontalSpacing(8)
        params_grid.setVerticalSpacing(8)
        params_grid.addWidget(w.maxiter_button, 0, 0)
        params_grid.addWidget(w.checkpointp_button, 0, 1)
        params_grid.addWidget(w.steps_button, 1, 0)
        params_grid.addWidget(w.testp_button, 1, 1)
        params_grid.setColumnStretch(0, 1)
        params_grid.setColumnStretch(1, 1)
        params_layout.addLayout(params_grid)

        readiness_card, readiness_layout = self._section(
            content,
            "READINESS",
            "Training stays blocked until every required input is ready.",
        )

        self.readiness_message = QLabel(readiness_card)
        self.readiness_message.setObjectName("trainReadinessMessage")
        self.readiness_message.setWordWrap(True)
        readiness_layout.addWidget(self.readiness_message)

        readiness_grid = QGridLayout()
        readiness_grid.setHorizontalSpacing(8)
        readiness_grid.setVerticalSpacing(8)
        self.ready_model = self._state_chip(readiness_card, "Model")
        self.ready_dataset = self._state_chip(readiness_card, "Dataset")
        self.ready_labels = self._state_chip(readiness_card, "Labels")
        self.ready_annotations = self._state_chip(readiness_card, "Annotations")
        readiness_grid.addWidget(self.ready_model, 0, 0)
        readiness_grid.addWidget(self.ready_dataset, 0, 1)
        readiness_grid.addWidget(self.ready_labels, 1, 0)
        readiness_grid.addWidget(self.ready_annotations, 1, 1)
        readiness_layout.addLayout(readiness_grid)
        w.validate_button.setToolTip(
            "Check model, dataset, labels and annotations are ready before training."
        )
        w.train_button.setToolTip(
            "Start training using the current configuration."
        )

        self.readiness_message.setToolTip(
            "Shows which requirements are complete before training can start."
        )

        readiness_layout.addWidget(w.validate_button, 0, Qt.AlignLeft)

        content_grid.addWidget(model_card, 0, 0)
        content_grid.addWidget(dataset_card, 0, 1)
        content_grid.addWidget(params_card, 1, 0)
        content_grid.addWidget(readiness_card, 1, 1)

        scroll.setWidget(content)
        self.root_layout.addWidget(scroll, 1)

        footer = QFrame(w)
        footer.setObjectName("trainFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(22, 12, 22, 12)
        footer_layout.setSpacing(14)

        footer_layout.addWidget(w.training_status_label, 1)
        footer_layout.addWidget(w.train_button, 0, Qt.AlignVCenter)
        self.root_layout.addWidget(footer)

    def _section(self, parent, title_text, subtitle_text):
        frame = QFrame(parent)
        frame.setObjectName("trainSectionCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title = QLabel(title_text, frame)
        title.setObjectName("trainSectionTitle")
        subtitle = QLabel(subtitle_text, frame)
        subtitle.setObjectName("trainSectionSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return frame, layout

    def _summary_row(self, parent):
        value = QLabel(parent)
        value.setObjectName("trainSummaryValue")
        value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        state = QLabel("• Checking", parent)
        state.setObjectName("trainSummaryState")
        state.setAlignment(Qt.AlignCenter)
        state.setMinimumWidth(78)
        return value, state

    def _row_with_label(self, label_text, value_widget, state_widget):
        row = QHBoxLayout()
        row.setSpacing(10)
        label = QLabel(label_text)
        label.setObjectName("trainFieldLabel")
        label.setMinimumWidth(58)
        row.addWidget(label)
        row.addWidget(value_widget, 1)
        row.addWidget(state_widget)
        return row

    def _state_chip(self, parent, name):
        chip = QLabel(parent)
        chip.setObjectName("trainReadinessChip")
        chip.setProperty("label", name)
        chip.setMinimumHeight(32)
        chip.setAlignment(Qt.AlignCenter)
        return chip

    def _connect_refresh_hooks(self):
        w = self.window
        for signal_owner in (
            w.p2_button,
            w.p3_button,
            w.list_button,
            w.dataset_button,
            w.label_button,
            w.generate_button,
            w.validate_button,
            w.maxiter_button,
            w.checkpointp_button,
            w.steps_button,
            w.testp_button,
            w.train_button,
        ):
            signal_owner.clicked.connect(self._sync_soon)
        w.model_selector.currentTextChanged.connect(self._sync_soon)

    def _sync_soon(self, *args):
        QTimer.singleShot(0, self.sync)
        QTimer.singleShot(400, self.sync)

    def _clear_legacy_inline_styles(self):
        w = self.window
        for widget in (
            w.p2_button,
            w.p3_button,
            w.model_selector,
            w.list_button,
            w.dataset_button,
            w.label_button,
            w.generate_button,
            w.validate_button,
            w.train_button,
        ):
            if widget.styleSheet():
                widget.setStyleSheet("")

    def sync(self):
        w = self.window
        self._clear_legacy_inline_styles()

        precision = int(getattr(w, "_precision_level", 2))
        self._set_selected(w.p2_button, precision == 2)
        self._set_selected(w.p3_button, precision == 3)

        model_name = str(getattr(w, "model_name", "")).strip()
        if not model_name:
            model_name = w.model_selector.currentText().strip() or "Not selected"
        dataset_path = str(getattr(w, "_path_to_dataset", "")).strip()
        labels_path = str(getattr(w, "_path_to_label_list", "")).strip()
        dataset_name = Path(dataset_path).name if dataset_path else "Not selected"
        labels_name = Path(labels_path).name if labels_path else "Not selected"

        model_ready = bool(getattr(w, "_is_model_ready", False))
        dataset_ready = bool(getattr(w, "_is_dataset_linked", False))
        labels_ready = bool(getattr(w, "_is_labellist_linked", False))
        annotations_ready = bool(getattr(w, "_is_dataset_labelled", False))

        self.model_value.setText(model_name)
        self.dataset_value.setText(dataset_name)
        self.dataset_value.setToolTip(dataset_path)
        self.labels_value.setText(labels_name)
        self.labels_value.setToolTip(labels_path)

        self._set_state(self.model_state, model_ready)
        self._set_state(self.dataset_state, dataset_ready)
        self._set_state(self.labels_state, labels_ready)
        self._set_readiness_chip(self.ready_model, "Model", model_ready)
        self._set_readiness_chip(self.ready_dataset, "Dataset", dataset_ready)
        self._set_readiness_chip(self.ready_labels, "Labels", labels_ready)
        self._set_readiness_chip(self.ready_annotations, "Annotations", annotations_ready)

        unmet = []
        if not model_ready:
            unmet.append("model")
        if not dataset_ready:
            unmet.append("dataset")
        if not labels_ready:
            unmet.append("label list")
        if not annotations_ready:
            unmet.append("labelled dataset")

        if unmet:
            self.readiness_message.setText(
                "Missing: " + ", ".join(unmet) + ".  Next: provide " + unmet[0] + "."
            )
        else:
            self.readiness_message.setText(
                "All prerequisites are ready. Validate once, then start training."
            )

        w.maxiter_button.setText(f"Max iterations  •  {w.max_iteration}")
        w.checkpointp_button.setText(f"Checkpoint  •  {w.checkpoint_period}")
        steps_text = str(w.steps).strip("()")
        if len(steps_text) > 26:
            steps_text = steps_text[:23] + "…"
        w.steps_button.setText(f"LR steps  •  {steps_text}")
        w.testp_button.setText(f"Test period  •  {w.test_period}")

        busy = bool(w._job_controller.is_busy())
        state_name = str(getattr(w._job_controller.state, "name", "")).upper()
        if state_name == "RUNNING" and getattr(w, "_active_operation", None) == "training":
            self.header_badge.setText("TRAINING")
            self.header_badge.setProperty("state", "running")
            w.train_button.setText("Training…")
        elif busy:
            self.header_badge.setText("BUSY")
            self.header_badge.setProperty("state", "running")
            w.train_button.setText("Train Model")
        elif not unmet:
            self.header_badge.setText("READY")
            self.header_badge.setProperty("state", "ready")
            w.train_button.setText("Train Model")
        else:
            self.header_badge.setText("SETUP REQUIRED")
            self.header_badge.setProperty("state", "blocked")
            w.train_button.setText("Train Model")

        self.header_badge.style().unpolish(self.header_badge)
        self.header_badge.style().polish(self.header_badge)

    def _set_selected(self, widget, selected):
        widget.setProperty("selected", selected)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _set_state(self, label, ready):
        label.setText("✓ Ready" if ready else "! Check")
        label.setProperty("state", "ready" if ready else "blocked")
        label.style().unpolish(label)
        label.style().polish(label)

    def _set_readiness_chip(self, label, name, ready):
        label.setText(f"✓ {name}" if ready else f"! {name}")
        label.setProperty("state", "ready" if ready else "blocked")
        label.style().unpolish(label)
        label.style().polish(label)

    def _apply_style(self):
        self.window.setStyleSheet(
            """
            QWidget#trainWindow,
            QWidget#trainContent {
                background-color: #101319;
                color: #e8edf5;
            }

            QFrame#trainHeader {
                background-color: #121720;
                border-bottom: 1px solid #27303d;
            }

            QLabel#trainEyebrow {
                color: #7e8ca3;
                font-size: 10px;
                font-weight: 700;
            }

            QLabel#trainTitle {
                color: #f7f9fc;
                font-size: 24px;
                font-weight: 700;
            }

            QLabel#trainSubtitle {
                color: #a9b4c4;
                font-size: 12px;
            }

            QLabel#trainHeaderBadge {
                color: #aab5c5;
                background-color: #1a202a;
                border: 1px solid #313b4a;
                border-radius: 8px;
                padding: 5px 10px;
                font-size: 10px;
                font-weight: 700;
            }

            QLabel#trainHeaderBadge[state="ready"] {
                color: #9de2b2;
                background-color: #14241b;
                border-color: #285b39;
            }

            QLabel#trainHeaderBadge[state="running"] {
                color: #b8c8ff;
                background-color: #17203a;
                border-color: #40599c;
            }

            QLabel#trainHeaderBadge[state="blocked"] {
                color: #e8c98f;
                background-color: #292116;
                border-color: #634a24;
            }

            QScrollArea#trainScroll,
            QScrollArea#trainScroll > QWidget > QWidget {
                background-color: #101319;
                border: none;
            }

            QFrame#trainSectionCard {
                background-color: #181d26;
                border: 1px solid #2b3441;
                border-radius: 12px;
            }

            QLabel#trainSectionTitle {
                color: #8c9ab0;
                font-size: 10px;
                font-weight: 700;
            }

            QLabel#trainSectionSubtitle {
                color: #758196;
                font-size: 11px;
            }

            QLabel#trainFieldLabel {
                color: #9ba7b8;
                font-size: 12px;
            }

            QLabel#trainSummaryValue {
                color: #e8edf5;
                font-size: 12px;
            }

            QLabel#trainSummaryState,
            QLabel#trainReadinessChip {
                color: #8793a5;
                background-color: #121720;
                border: 1px solid #2d3644;
                border-radius: 7px;
                padding: 4px 7px;
                font-size: 10px;
                font-weight: 600;
            }

            QLabel#trainSummaryState[state="ready"],
            QLabel#trainReadinessChip[state="ready"] {
                color: #98dfad;
                background-color: #14231a;
                border-color: #2b5b3a;
            }

            QLabel#trainSummaryState[state="blocked"],
            QLabel#trainReadinessChip[state="blocked"] {
                color: #e4c28a;
                background-color: #261f15;
                border-color: #5c4626;
            }

            QLabel#trainReadinessMessage {
                color: #b5c0cf;
                background-color: #121720;
                border: 1px solid #293240;
                border-radius: 8px;
                padding: 8px 10px;
                font-size: 11px;
            }

            QPushButton,
            QComboBox {
                color: #dce3ed;
                background-color: #121720;
                border: 1px solid #303a49;
                border-radius: 7px;
                padding: 6px 9px;
                font-size: 12px;
            }

            QPushButton:hover,
            QComboBox:hover {
                background-color: #1c2330;
                border-color: #536278;
            }

            QPushButton:pressed {
                background-color: #151b24;
            }

            QPushButton:disabled,
            QComboBox:disabled {
                color: #626e80;
                background-color: #141820;
                border-color: #252c37;
            }

            QPushButton#precisionButton {
                min-width: 56px;
                font-weight: 700;
            }

            QPushButton#precisionButton[selected="true"] {
                color: #dce5ff;
                background-color: #202b45;
                border-color: #5369a3;
            }

            QComboBox QAbstractItemView {
                color: #e3e8ef;
                background-color: #171c24;
                border: 1px solid #354052;
                selection-background-color: #28354b;
            }

            QPushButton#validateAction {
                color: #cbd5e6;
                background-color: #171d27;
                border-color: #344154;
                padding: 7px 13px;
            }

            QFrame#trainFooter {
                background-color: #121720;
                border-top: 1px solid #27303d;
            }

            QLabel#trainRuntimeStatus {
                color: #aab5c5;
                font-size: 11px;
            }

            QPushButton#trainPrimary {
                color: #ffffff;
                background-color: #5367d8;
                border: 1px solid #6e80e6;
                border-radius: 9px;
                padding: 9px 18px;
                font-size: 13px;
                font-weight: 700;
            }

            QPushButton#trainPrimary:hover {
                background-color: #6275e2;
                border-color: #8796ee;
            }

            QPushButton#trainPrimary:disabled {
                color: #727b90;
                background-color: #1c2230;
                border-color: #2d3545;
            }
            """
        )

        # Object names are assigned before polishing so the primary/secondary
        # action styles are active immediately on first render.
        for widget in (self.window.train_button, self.window.validate_button):
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def eventFilter(self, obj, event):
        if obj is self.window and event.type() in (QEvent.Hide, QEvent.Close):
            if not self.window.isMaximized():
                self.settings.setValue(self.SETTINGS_KEY, self.window.size())
        return super().eventFilter(obj, event)


def apply_train_ui_refresh(window):
    """Apply a presentation-only refresh to an existing TrainWindow."""
    controller = _TrainUiController(window)
    controller.apply()
    window._train_ui_controller = controller
    return controller
