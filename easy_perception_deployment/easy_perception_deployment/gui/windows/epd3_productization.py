"""EPD-3 productization: Smart Model Manager and deployment preflight gate."""

from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from windows.job_controller import JobState
from windows.model_manager import (
    inspect_deployment_model,
    library_install_state,
    load_model_library,
)


class _InspectionSignals(QObject):
    finished = Signal(int, dict)


class _InspectionWorker(QObject):
    """Run file hashing and ONNX Runtime inspection away from the Qt GUI thread."""

    def __init__(
        self,
        generation,
        model_path,
        label_path,
        usecase_mode,
        package_root,
    ):
        super().__init__()
        self.generation = generation
        self.model_path = model_path
        self.label_path = label_path
        self.usecase_mode = usecase_mode
        self.package_root = package_root
        self.signals = _InspectionSignals()

    @Slot()
    def run(self):
        try:
            result = inspect_deployment_model(
                self.model_path,
                self.label_path,
                self.usecase_mode,
                self.package_root,
            )
        except Exception as exc:
            result = {
                "status": "blocked",
                "summary": f"Model inspection failed unexpectedly: {exc}",
                "model_path": self.model_path,
                "label_path": self.label_path,
                "mode": self.usecase_mode,
                "mode_name": "Unknown",
                "blockers": [f"Model inspection failed unexpectedly: {exc}"],
                "warnings": [],
                "inspection_source": "none",
            }
        self.signals.finished.emit(self.generation, result)


class ModelManagerDialog(QDialog):
    """Explain the current model and expose the trusted pretrained catalog."""

    def __init__(self, controller):
        super().__init__(controller.deploy)
        self.controller = controller
        self.deploy = controller.deploy
        self.package_root = Path(self.deploy._PACKAGE_ROOT)
        self.library = []
        self._library_rows = []

        self.setWindowTitle("EPD Smart Model Manager")
        self.resize(960, 700)
        self.setMinimumSize(760, 560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(12)

        title = QLabel("Smart Model Manager", self)
        title.setObjectName("modelManagerTitle")
        subtitle = QLabel(
            "Inspect ONNX compatibility, labels and perception-mode fit before Run.",
            self,
        )
        subtitle.setObjectName("modelManagerSubtitle")
        subtitle.setWordWrap(True)
        outer.addWidget(title)
        outer.addWidget(subtitle)

        self.tabs = QTabWidget(self)
        outer.addWidget(self.tabs, 1)

        self._build_current_tab()
        self._build_library_tab()

        footer = QHBoxLayout()
        self.refresh_button = QPushButton("Inspect again", self)
        self.refresh_button.clicked.connect(self.controller.schedule_inspection)
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.close)
        footer.addWidget(self.refresh_button)
        footer.addStretch(1)
        footer.addWidget(close_button)
        outer.addLayout(footer)

        self._apply_style()
        self.refresh_library()

    def _build_current_tab(self):
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.current_status = QLabel("CHECKING", tab)
        self.current_status.setObjectName("managerStatus")
        self.current_status.setAlignment(Qt.AlignCenter)
        self.current_status.setMinimumHeight(34)
        layout.addWidget(self.current_status, 0, Qt.AlignLeft)

        self.current_summary = QLabel("Inspecting selected model…", tab)
        self.current_summary.setObjectName("managerSummary")
        self.current_summary.setWordWrap(True)
        layout.addWidget(self.current_summary)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        self.model_value = self._value_label(tab)
        self.task_value = self._value_label(tab)
        self.io_value = self._value_label(tab)
        self.labels_value = self._value_label(tab)
        self.mode_value = self._value_label(tab)
        self.source_value = self._value_label(tab)
        rows = [
            ("Model", self.model_value),
            ("Task / EPD level", self.task_value),
            ("ONNX I/O", self.io_value),
            ("Labels", self.labels_value),
            ("Mode compatibility", self.mode_value),
            ("Inspection source", self.source_value),
        ]
        for row, (name, value) in enumerate(rows):
            label = QLabel(name, tab)
            label.setObjectName("managerField")
            grid.addWidget(label, row, 0, Qt.AlignTop)
            grid.addWidget(value, row, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self.current_details = QTextBrowser(tab)
        self.current_details.setObjectName("managerDetails")
        self.current_details.setOpenExternalLinks(True)
        layout.addWidget(self.current_details, 1)
        self.tabs.addTab(tab, "Current model")

    def _build_library_tab(self):
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        intro = QLabel(
            "Trusted catalog entries are identified by exact SHA256, not filename. "
            "Missing models can still be downloaded from their upstream source.",
            tab,
        )
        intro.setWordWrap(True)
        intro.setObjectName("modelManagerSubtitle")
        layout.addWidget(intro)

        self.library_table = QTableWidget(0, 5, tab)
        self.library_table.setHorizontalHeaderLabels(
            ["Model", "Installed", "EPD", "Task", "Recommended modes"]
        )
        self.library_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.library_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.library_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.library_table.verticalHeader().setVisible(False)
        self.library_table.horizontalHeader().setStretchLastSection(True)
        self.library_table.itemSelectionChanged.connect(self._library_selection_changed)
        layout.addWidget(self.library_table, 1)

        self.library_details = QTextBrowser(tab)
        self.library_details.setObjectName("libraryDetails")
        self.library_details.setMaximumHeight(160)
        self.library_details.setOpenExternalLinks(True)
        layout.addWidget(self.library_details)

        actions = QHBoxLayout()
        self.use_library_button = QPushButton("Use model + matching labels", tab)
        self.use_library_button.clicked.connect(self._use_selected_library_model)
        self.open_source_button = QPushButton("Open upstream source", tab)
        self.open_source_button.clicked.connect(self._open_selected_source)
        actions.addWidget(self.use_library_button)
        actions.addWidget(self.open_source_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.tabs.addTab(tab, "Pretrained library")

    @staticmethod
    def _value_label(parent):
        label = QLabel("—", parent)
        label.setObjectName("managerValue")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return label

    def show_manager(self):
        self.refresh_library()
        self.set_result(self.controller.result)
        self.show()
        self.raise_()
        self.activateWindow()

    def set_checking(self):
        self.current_status.setText("CHECKING")
        self.current_status.setProperty("state", "checking")
        self.current_summary.setText("Inspecting the selected ONNX model and labels…")
        self._repolish(self.current_status)

    def set_result(self, result):
        if not result:
            self.set_checking()
            return

        status = result.get("status", "blocked")
        self.current_status.setText("READY" if status == "ready" else "BLOCKED")
        self.current_status.setProperty("state", status)
        self._repolish(self.current_status)
        self.current_summary.setText(result.get("summary", ""))

        model_path = str(result.get("model_path", "") or "")
        entry = result.get("library_entry") or {}
        trusted_name = entry.get("name")
        self.model_value.setText(trusted_name or Path(model_path).name or "Not configured")
        self.model_value.setToolTip(model_path)

        precision = result.get("precision_level")
        precision_text = "Unknown" if precision is None else f"P{precision}"
        self.task_value.setText(f"{precision_text} • {result.get('task', 'Unknown')}")

        metadata = result.get("metadata") or {}
        input_count = metadata.get("input_count")
        output_count = metadata.get("output_count")
        if input_count is None or output_count is None:
            self.io_value.setText("Not inspected")
        else:
            self.io_value.setText(f"{input_count} input(s) • {output_count} output(s)")

        labels = result.get("labels") or {}
        self.labels_value.setText(labels.get("detail", "Not validated"))

        supported = result.get("supported_modes", []) or []
        selected = result.get("mode_name", "Unknown")
        supported_text = ", ".join(supported) if supported else "none"
        self.mode_value.setText(f"Selected: {selected} • Supported: {supported_text}")
        self.source_value.setText(result.get("inspection_source", "none"))

        blockers = result.get("blockers", []) or []
        warnings = result.get("warnings", []) or []
        recommended = result.get("recommended_modes", []) or []
        digest = result.get("sha256", "")

        sections = []
        if blockers:
            items = "".join(f"<li>{html.escape(item)}</li>" for item in blockers)
            sections.append(f"<h3>Blocking incompatibilities</h3><ul>{items}</ul>")
        if warnings:
            items = "".join(f"<li>{html.escape(item)}</li>" for item in warnings)
            sections.append(f"<h3>Warnings / limits</h3><ul>{items}</ul>")
        if recommended:
            sections.append(
                "<h3>Recommended perception modes</h3><p>"
                + html.escape(", ".join(recommended))
                + "</p>"
            )
        if digest:
            sections.append(
                "<h3>Model identity</h3><p><b>SHA256:</b> <code>"
                + html.escape(digest)
                + "</code></p>"
            )
        if not sections:
            sections.append("<p>No additional model details are available yet.</p>")
        self.current_details.setHtml("".join(sections))

    def refresh_library(self):
        self.library = load_model_library(self.package_root)
        self._library_rows = []
        self.library_table.setRowCount(len(self.library))

        for row, entry in enumerate(self.library):
            state, path = library_install_state(self.package_root, entry)
            self._library_rows.append((entry, state, path))
            values = [
                entry.get("name", entry.get("filename", "Unknown")),
                state,
                f"P{entry.get('precision_level', '?')}",
                entry.get("task", "Unknown"),
                ", ".join(entry.get("recommended_modes", []) or []) or "Not supported",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                self.library_table.setItem(row, column, item)

        self.library_table.resizeColumnsToContents()
        if self.library:
            self.library_table.selectRow(0)
        else:
            self.library_details.setText("Trusted model catalog is unavailable.")
            self.use_library_button.setEnabled(False)
            self.open_source_button.setEnabled(False)

    def _selected_library(self):
        row = self.library_table.currentRow()
        if row < 0 or row >= len(self._library_rows):
            return None
        return self._library_rows[row]

    def _library_selection_changed(self):
        selected = self._selected_library()
        if selected is None:
            self.use_library_button.setEnabled(False)
            self.open_source_button.setEnabled(False)
            return

        entry, state, path = selected
        precision = int(entry.get("precision_level", 0) or 0)
        self.use_library_button.setEnabled(state == "installed" and precision in (2, 3))
        self.open_source_button.setEnabled(bool(entry.get("source_url")))

        notes = html.escape(str(entry.get("notes", "")))
        source = html.escape(str(entry.get("source_url", "")))
        labels = entry.get("canonical_labels") or "Model-specific labels required"
        detail = (
            f"<p><b>{html.escape(str(entry.get('name', 'Model')))}</b></p>"
            f"<p>Status: <b>{html.escape(state)}</b><br>"
            f"Local path: <code>{html.escape(path)}</code><br>"
            f"Labels: {html.escape(str(labels))}</p>"
            f"<p>{notes}</p>"
            f"<p>Source: <code>{source}</code></p>"
        )
        self.library_details.setHtml(detail)

    def _use_selected_library_model(self):
        selected = self._selected_library()
        if selected is None:
            return
        entry, state, model_path = selected
        if state != "installed":
            return
        self.controller.use_library_model(entry, model_path)
        self.tabs.setCurrentIndex(0)

    def _open_selected_source(self):
        selected = self._selected_library()
        if selected is None:
            return
        entry, _, _ = selected
        url = str(entry.get("source_url", "") or "")
        if url:
            QDesktopServices.openUrl(QUrl(url))

    @staticmethod
    def _repolish(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QDialog {
                background-color: #101319;
                color: #e8edf5;
            }
            QLabel#modelManagerTitle {
                color: #f7f9fc;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#modelManagerSubtitle {
                color: #95a2b5;
                font-size: 11px;
            }
            QLabel#managerStatus {
                color: #d6bd82;
                background-color: #292116;
                border: 1px solid #634a24;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#managerStatus[state="ready"] {
                color: #9de2b2;
                background-color: #14241b;
                border-color: #285b39;
            }
            QLabel#managerStatus[state="blocked"] {
                color: #e7a4a4;
                background-color: #2b171a;
                border-color: #69353d;
            }
            QLabel#managerSummary {
                color: #dce3ed;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#managerField {
                color: #8896a9;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#managerValue {
                color: #dce3ed;
                font-size: 11px;
            }
            QTabWidget::pane,
            QTextBrowser,
            QTableWidget {
                background-color: #151a22;
                border: 1px solid #2c3644;
                border-radius: 8px;
            }
            QHeaderView::section {
                color: #aab6c7;
                background-color: #1b222d;
                border: none;
                border-right: 1px solid #303a48;
                padding: 6px;
                font-weight: 600;
            }
            QPushButton {
                color: #dce3ed;
                background-color: #1a202a;
                border: 1px solid #354155;
                border-radius: 7px;
                padding: 7px 11px;
            }
            QPushButton:hover {
                background-color: #232c39;
            }
            QPushButton:disabled {
                color: #657185;
                background-color: #151a22;
                border-color: #292f39;
            }
            """
        )


class _EPD3Controller(QObject):
    """Attach automatic model truth to the existing Deploy window."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.deploy = main_window.deploy_window
        self.result = None
        self.state = "checking"
        self._generation = 0
        self._worker_thread = None
        self._worker = None
        self._inspection_active = False
        self._inspection_pending = False

        self.status_badge = None
        self.summary_label = None
        self.manager_button = None
        self.dialog = ModelManagerDialog(self)

        self._install_model_card_controls()
        self._connect_hooks()

        self.gate_timer = QTimer(self)
        self.gate_timer.setInterval(300)
        self.gate_timer.timeout.connect(self._enforce_run_gate)
        self.gate_timer.start()
        self.schedule_inspection()

    def _install_model_card_controls(self):
        model_card = None
        for label in self.deploy.findChildren(QLabel):
            if label.objectName() == "sectionTitle" and label.text().strip() == "MODEL":
                parent = label.parentWidget()
                if isinstance(parent, QFrame):
                    model_card = parent
                    break
        if model_card is None or model_card.layout() is None:
            return

        row = QHBoxLayout()
        row.setSpacing(8)
        caption = QLabel("Model check", model_card)
        caption.setObjectName("modelTruthCaption")
        self.status_badge = QLabel("CHECKING", model_card)
        self.status_badge.setObjectName("modelTruthBadge")
        self.status_badge.setProperty("modelState", "checking")
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setMinimumWidth(82)
        self.manager_button = QPushButton("Model Manager", model_card)
        self.manager_button.setObjectName("modelManagerButton")
        self.manager_button.setToolTip(
            "Inspect ONNX validity, EPD precision level, labels and compatible modes."
        )
        self.manager_button.clicked.connect(self.dialog.show_manager)
        row.addWidget(caption)
        row.addWidget(self.status_badge)
        row.addStretch(1)
        row.addWidget(self.manager_button)
        model_card.layout().addLayout(row)

        self.summary_label = QLabel("Inspecting model compatibility…", model_card)
        self.summary_label.setObjectName("modelTruthSummary")
        self.summary_label.setWordWrap(True)
        model_card.layout().addWidget(self.summary_label)
        self._append_style()

    def _connect_hooks(self):
        for button in (
            self.deploy.model_button,
            self.deploy.list_button,
            self.deploy.use_defaults_button,
        ):
            button.clicked.connect(self.schedule_inspection)
        self.deploy.usecase_config_button.currentTextChanged.connect(
            self.schedule_inspection
        )

    @Slot()
    def schedule_inspection(self, *args):
        self._inspection_pending = True
        self.state = "checking"
        self.result = None
        self._set_compact_state("CHECKING", "checking", "Inspecting ONNX model…")
        self.dialog.set_checking()
        self._enforce_run_gate()
        QTimer.singleShot(0, self._start_inspection)

    def _start_inspection(self):
        if self._inspection_active or not self._inspection_pending:
            return

        self._inspection_pending = False
        self._inspection_active = True
        self._generation += 1
        generation = self._generation

        model_path = self.deploy.resolveFilePath(self.deploy._path_to_model)
        label_path = self.deploy.resolveFilePath(self.deploy._path_to_label_list)
        usecase_mode = int(getattr(self.deploy, "usecase_mode", 0))
        package_root = str(self.deploy._PACKAGE_ROOT)

        thread = QThread(self)
        worker = _InspectionWorker(
            generation,
            model_path,
            label_path,
            usecase_mode,
            package_root,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.signals.finished.connect(self._inspection_finished)
        worker.signals.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker_refs)
        self._worker_thread = thread
        self._worker = worker
        thread.start()

    @Slot(int, dict)
    def _inspection_finished(self, generation, result):
        if generation != self._generation or self._inspection_pending:
            return

        self.result = result
        self.state = result.get("status", "blocked")
        if self.state == "ready":
            precision = result.get("precision_level")
            task = result.get("task", "Compatible model")
            summary = f"P{precision} • {task} • {result.get('mode_name', '')} ready"
            self._set_compact_state("READY", "ready", summary)
        else:
            summary = result.get("summary", "Model is incompatible.")
            self._set_compact_state("BLOCKED", "blocked", summary)

        self.dialog.set_result(result)
        self.deploy.validateDeployInputs()
        self._enforce_run_gate()
        controller = getattr(self.main_window, "_deploy_ui_controller", None)
        if controller is not None:
            controller.sync()
        self._enforce_run_gate()

    @Slot()
    def _clear_worker_refs(self):
        self._worker = None
        self._worker_thread = None
        self._inspection_active = False
        if self._inspection_pending:
            QTimer.singleShot(0, self._start_inspection)

    def _set_compact_state(self, text, state, summary):
        if self.status_badge is not None:
            self.status_badge.setText(text)
            self.status_badge.setProperty("modelState", state)
            self.status_badge.style().unpolish(self.status_badge)
            self.status_badge.style().polish(self.status_badge)
        if self.summary_label is not None:
            self.summary_label.setText(summary)

    def _enforce_run_gate(self):
        job_state = self.deploy._job_controller.state
        if job_state in (JobState.STARTING, JobState.RUNNING, JobState.STOPPING):
            return

        if self.state == "checking":
            self.deploy.run_button.setEnabled(False)
            message = "Run disabled: Smart Model Manager is inspecting the selected model."
            self.deploy.run_button.setToolTip(message)
            if self.deploy.validation_label.text().startswith("Run enabled"):
                self.deploy.validation_label.setText(message)
            return

        if self.state != "ready":
            message = "Run disabled: model compatibility check failed."
            if self.result:
                message = "Run disabled: " + str(
                    self.result.get("summary", "model compatibility check failed.")
                )
            self.deploy.run_button.setEnabled(False)
            self.deploy.run_button.setToolTip(message)
            self.deploy.validation_label.setText(message)

    def use_library_model(self, entry, model_path):
        model_path = Path(model_path)
        self.deploy._path_to_model = self.deploy._normalize_data_path(str(model_path))

        canonical_labels = entry.get("canonical_labels")
        if canonical_labels:
            label_path = (
                Path(self.deploy._PACKAGE_ROOT)
                / "data"
                / "label_list"
                / str(canonical_labels)
            )
            if label_path.is_file():
                self.deploy._path_to_label_list = self.deploy._normalize_data_path(
                    str(label_path)
                )

        self.deploy.updateSessionConfig()
        self.deploy.validateDeployInputs()
        controller = getattr(self.main_window, "_deploy_ui_controller", None)
        if controller is not None:
            controller.sync()
        self.schedule_inspection()

    def _append_style(self):
        self.deploy.setStyleSheet(
            self.deploy.styleSheet()
            + """
            QLabel#modelTruthCaption {
                color: #8d99aa;
                font-size: 10px;
            }
            QLabel#modelTruthBadge {
                color: #e3c98f;
                background-color: #292116;
                border: 1px solid #634a24;
                border-radius: 7px;
                padding: 4px 8px;
                font-size: 9px;
                font-weight: 700;
            }
            QLabel#modelTruthBadge[modelState="ready"] {
                color: #9de2b2;
                background-color: #14241b;
                border-color: #285b39;
            }
            QLabel#modelTruthBadge[modelState="blocked"] {
                color: #e7a4a4;
                background-color: #2b171a;
                border-color: #69353d;
            }
            QLabel#modelTruthSummary {
                color: #8390a3;
                font-size: 9px;
            }
            QPushButton#modelManagerButton {
                color: #cbd5e3;
                background-color: #171d26;
                border: 1px solid #354155;
                border-radius: 7px;
                padding: 5px 9px;
                font-size: 10px;
            }
            """
        )


def apply_epd3_productization(main_window):
    """Install EPD-3 Smart Model Manager once on the existing main window."""
    if getattr(main_window, "_epd3_productization_applied", False):
        return None
    main_window._epd3_productization_applied = True
    controller = _EPD3Controller(main_window)
    main_window._epd3_model_manager_controller = controller
    return controller
