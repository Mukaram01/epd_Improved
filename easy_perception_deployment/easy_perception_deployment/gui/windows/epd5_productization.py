"""EPD-5 productization: reusable perception profiles and replay workflows."""

from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from windows.job_controller import JobState
from windows.profile_store import (
    ProfileError,
    ProfileStore,
    apply_profile_to_files,
    capture_profile,
    profile_summary,
)

_TOPIC_RE = re.compile(r"Topic:\s*([^\s|]+)")


def parse_rosbag_topics(text):
    """Extract topic names from common `ros2 bag info` output formats."""
    topics = set()
    for line in str(text or "").splitlines():
        match = _TOPIC_RE.search(line)
        if match:
            topics.add(match.group(1).strip())
            continue
        stripped = line.strip()
        if stripped.startswith("/"):
            topics.add(stripped.split()[0].rstrip(":"))
    return sorted(topics)


def replay_command(fixture, mode, summary_path):
    """Build the deterministic replay launch command without shell quoting."""
    return [
        "ros2",
        "launch",
        "easy_perception_deployment",
        "replay.launch.py",
        f"fixture:={Path(fixture).expanduser()}",
        f"mode:={mode}",
        f"summary_output:={Path(summary_path).expanduser()}",
    ]


def rosbag_play_command(bag_path):
    return ["ros2", "bag", "play", str(Path(bag_path).expanduser())]


class ProfilesReplayDialog(QDialog):
    """Combined operator surface for EPD profiles and deterministic replay."""

    def __init__(self, controller):
        super().__init__(controller.deploy)
        self.controller = controller
        self.setWindowTitle("EPD Profiles & Replay")
        self.resize(1040, 720)
        self.setMinimumSize(800, 600)

        outer = QVBoxLayout(self)
        title = QLabel("Profiles & Replay", self)
        title.setObjectName("epd5Title")
        subtitle = QLabel(
            "Save known-good perception configurations and reproduce them with fixtures or rosbag.",
            self,
        )
        subtitle.setWordWrap(True)
        outer.addWidget(title)
        outer.addWidget(subtitle)

        self.tabs = QTabWidget(self)
        outer.addWidget(self.tabs, 1)
        self._build_profiles_tab()
        self._build_replay_tab()

        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.close)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(close_button)
        outer.addLayout(footer)
        self._apply_style()

    def _build_profiles_tab(self):
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)
        intro = QLabel(
            "A profile captures model, labels, RGB topic, perception mode and runtime settings. "
            "Asset hashes prevent a different model from being substituted silently.",
            tab,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.profile_table = QTableWidget(0, 6, tab)
        self.profile_table.setHorizontalHeaderLabels(
            ["Name", "Mode", "Model", "Camera topic", "Device", "Known-good"]
        )
        self.profile_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.profile_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.profile_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.profile_table.verticalHeader().setVisible(False)
        self.profile_table.horizontalHeader().setStretchLastSection(True)
        self.profile_table.itemSelectionChanged.connect(self.controller.profile_selection_changed)
        layout.addWidget(self.profile_table, 1)

        self.profile_details = QTextBrowser(tab)
        self.profile_details.setMaximumHeight(150)
        layout.addWidget(self.profile_details)

        actions = QGridLayout()
        save = QPushButton("Save current", tab)
        save.clicked.connect(self.controller.save_current_profile)
        apply_button = QPushButton("Apply selected", tab)
        apply_button.clicked.connect(self.controller.apply_selected_profile)
        known = QPushButton("Set known-good", tab)
        known.clicked.connect(self.controller.set_selected_known_good)
        restore = QPushButton("Restore known-good", tab)
        restore.clicked.connect(self.controller.restore_known_good)
        import_button = QPushButton("Import", tab)
        import_button.clicked.connect(self.controller.import_profile)
        export_button = QPushButton("Export", tab)
        export_button.clicked.connect(self.controller.export_profile)
        delete_button = QPushButton("Delete", tab)
        delete_button.clicked.connect(self.controller.delete_profile)
        refresh = QPushButton("Refresh", tab)
        refresh.clicked.connect(self.controller.refresh_profiles)
        buttons = [
            save,
            apply_button,
            known,
            restore,
            import_button,
            export_button,
            delete_button,
            refresh,
        ]
        for index, button in enumerate(buttons):
            actions.addWidget(button, index // 4, index % 4)
        layout.addLayout(actions)

        self.profile_status = QLabel("No profile applied in this session.", tab)
        self.profile_status.setWordWrap(True)
        layout.addWidget(self.profile_status)
        self.tabs.addTab(tab, "Perception profiles")

    def _build_replay_tab(self):
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)

        deterministic = QLabel("Deterministic fixture replay", tab)
        deterministic.setObjectName("epd5Section")
        layout.addWidget(deterministic)
        explain = QLabel(
            "Runs the existing EPD replay.launch.py pipeline using recorded RGB/depth/CameraInfo "
            "fixture observations and produces a PASS/FAIL acceptance summary.",
            tab,
        )
        explain.setWordWrap(True)
        layout.addWidget(explain)

        fixture_row = QHBoxLayout()
        self.fixture_path = QLineEdit(tab)
        self.fixture_path.setPlaceholderText("Select a replay fixture JSON")
        browse_fixture = QPushButton("Browse fixture", tab)
        browse_fixture.clicked.connect(self.controller.choose_fixture)
        fixture_row.addWidget(self.fixture_path, 1)
        fixture_row.addWidget(browse_fixture)
        layout.addLayout(fixture_row)

        fixture_actions = QHBoxLayout()
        self.fixture_mode = QComboBox(tab)
        self.fixture_mode.addItems(["fast", "realtime"])
        run_fixture = QPushButton("Run deterministic replay", tab)
        run_fixture.clicked.connect(self.controller.run_fixture_replay)
        fixture_actions.addWidget(QLabel("Mode", tab))
        fixture_actions.addWidget(self.fixture_mode)
        fixture_actions.addWidget(run_fixture)
        fixture_actions.addStretch(1)
        layout.addLayout(fixture_actions)

        bag_title = QLabel("ROS bag replay", tab)
        bag_title.setObjectName("epd5Section")
        layout.addWidget(bag_title)
        bag_explain = QLabel(
            "Use a bag when it publishes the same topic names expected by the active profile. "
            "Start Deploy first; EPD-5 does not silently remap recorded topics.",
            tab,
        )
        bag_explain.setWordWrap(True)
        layout.addWidget(bag_explain)

        bag_row = QHBoxLayout()
        self.bag_path = QLineEdit(tab)
        self.bag_path.setPlaceholderText("Select rosbag2 directory")
        browse_bag = QPushButton("Browse bag", tab)
        browse_bag.clicked.connect(self.controller.choose_bag)
        bag_row.addWidget(self.bag_path, 1)
        bag_row.addWidget(browse_bag)
        layout.addLayout(bag_row)

        bag_actions = QHBoxLayout()
        inspect_bag = QPushButton("Inspect bag", tab)
        inspect_bag.clicked.connect(self.controller.inspect_bag)
        play_bag = QPushButton("Play bag", tab)
        play_bag.clicked.connect(self.controller.play_bag)
        stop = QPushButton("Stop replay", tab)
        stop.clicked.connect(self.controller.stop_replay)
        bag_actions.addWidget(inspect_bag)
        bag_actions.addWidget(play_bag)
        bag_actions.addWidget(stop)
        bag_actions.addStretch(1)
        layout.addLayout(bag_actions)

        self.replay_state = QLabel("IDLE", tab)
        self.replay_state.setObjectName("epd5State")
        layout.addWidget(self.replay_state, 0, Qt.AlignLeft)
        self.replay_details = QTextBrowser(tab)
        layout.addWidget(self.replay_details, 1)
        self.tabs.addTab(tab, "Replay")

    def selected_profile_path(self):
        row = self.profile_table.currentRow()
        if row < 0:
            return None
        item = self.profile_table.item(row, 0)
        return item.data(Qt.UserRole) if item is not None else None

    def _apply_style(self):
        self.setStyleSheet(
            """
            QDialog { background: #15191f; color: #e8edf2; }
            QLabel#epd5Title { font-size: 24px; font-weight: 700; }
            QLabel#epd5Section { font-size: 16px; font-weight: 700; margin-top: 8px; }
            QLabel#epd5State {
                background: #29313a; border-radius: 6px; padding: 6px 10px;
                font-weight: 700;
            }
            QTableWidget, QTextBrowser, QLineEdit {
                background: #11161b; border: 1px solid #303944; border-radius: 6px;
            }
            QPushButton { min-height: 32px; padding: 0 10px; }
            """
        )


class EPD5Controller(QObject):
    """Own profiles/replay without changing EPD runtime or scene ownership."""

    def __init__(self, main_window, store=None):
        super().__init__(main_window.deploy_window)
        self.main_window = main_window
        self.deploy = main_window.deploy_window
        self.package_root = Path(self.deploy._PACKAGE_ROOT)
        self.store = store or ProfileStore()
        self.dialog = ProfilesReplayDialog(self)
        self.process = None
        self.process_kind = None
        self.summary_path = Path("/tmp/epd_profile_replay_summary.json")
        self.events = queue.Queue()
        self.worker = None
        self.active_profile = None

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(250)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.start()
        self._install_button()
        self._set_default_fixture()
        self.refresh_profiles()

    def _install_button(self):
        button = QPushButton("Profiles & Replay", self.deploy)
        button.setToolTip(
            "Save or restore reproducible perception profiles and replay recorded inputs."
        )
        button.clicked.connect(self.show)
        self.deploy.profiles_replay_button = button
        ui = getattr(self.main_window, "_deploy_ui_controller", None)
        badge = getattr(ui, "header_badge", None)
        header = badge.parentWidget() if badge is not None else None
        if header is not None and header.layout() is not None:
            index = max(0, header.layout().count() - 1)
            header.layout().insertWidget(index, button, 0, Qt.AlignTop)
        else:
            self.deploy.layout().addWidget(button)

    def _set_default_fixture(self):
        fixture = self.package_root / "fixtures" / "p8_tracking.json"
        if fixture.is_file():
            self.dialog.fixture_path.setText(str(fixture))

    def show(self):
        self.refresh_profiles()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def refresh_profiles(self):
        records = self.store.list_profiles()
        known_good = self.store.known_good_path()
        self.dialog.profile_table.setRowCount(len(records))
        for row, (path, profile) in enumerate(records):
            summary = profile_summary(profile)
            known = bool(
                known_good is not None
                and path.resolve() == known_good.resolve()
            )
            values = [
                summary["name"],
                summary["mode"],
                summary["model"],
                summary["topic"],
                summary["device"],
                "✓" if known else "",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.UserRole, str(path))
                self.dialog.profile_table.setItem(row, column, item)
        self.dialog.profile_table.resizeColumnsToContents()
        return records

    def profile_selection_changed(self):
        path = self.dialog.selected_profile_path()
        if not path:
            self.dialog.profile_details.clear()
            return
        try:
            profile = self.store.load(path)
            summary = profile_summary(profile)
        except ProfileError as exc:
            self.dialog.profile_details.setPlainText(str(exc))
            return
        details = [
            f"Name: {summary['name']}",
            f"Mode: {summary['mode']}",
            f"Model: {summary['model']}",
            f"Labels: {summary['labels']}",
            f"Camera: {summary['topic']}",
            f"Device: {summary['device']}",
            f"Confidence: {summary['confidence']:.2f}",
            f"Transport: {summary['transport']}",
            f"Detection overlay: {'On' if summary['overlay'] else 'Off'}",
            f"Object masks: {'On' if summary['masks'] else 'Off'}",
        ]
        if summary["description"]:
            details.extend(["", summary["description"]])
        self.dialog.profile_details.setPlainText("\n".join(details))

    def save_current_profile(self):
        name, ok = QInputDialog.getText(
            self.dialog,
            "Save perception profile",
            "Profile name:",
        )
        if not ok or not name.strip():
            return
        description, accepted = QInputDialog.getMultiLineText(
            self.dialog,
            "Profile description",
            "Optional notes (workcell, camera position, part family, acceptance state):",
        )
        if not accepted:
            description = ""
        try:
            profile = capture_profile(
                self.package_root,
                name.strip(),
                description,
            )
            path = self.store.save(profile)
        except ProfileError as exc:
            self._error("Unable to save profile", exc)
            return
        self.active_profile = str(path)
        self.dialog.profile_status.setText(
            f"Saved current Deploy configuration as {profile['name']}."
        )
        self.refresh_profiles()

    def apply_selected_profile(self):
        path = self.dialog.selected_profile_path()
        if not path:
            self.dialog.profile_status.setText("Select a profile first.")
            return
        self._apply_profile_path(Path(path))

    def _apply_profile_path(self, path):
        if self.deploy._job_controller.state in (
            JobState.STARTING,
            JobState.RUNNING,
            JobState.STOPPING,
        ):
            self.dialog.profile_status.setText(
                "Stop the active deployment before changing perception profile."
            )
            return False
        try:
            profile = self.store.load(path)
            applied, warnings = apply_profile_to_files(
                profile,
                self.package_root,
            )
            self._sync_deploy(applied)
        except ProfileError as exc:
            self._error("Unable to apply profile", exc)
            return False
        self.active_profile = str(path)
        message = f"Applied profile: {applied['name']}"
        if warnings:
            message += " — " + "; ".join(warnings)
        self.dialog.profile_status.setText(message)
        return True

    def _sync_deploy(self, profile):
        epd = profile["epd"]
        session = epd["session_config"]
        usecase = epd["usecase_config"]
        input_topic = epd["input_image_topic_config"]
        deploy = self.deploy

        deploy._path_to_model = str(session["path_to_model"])
        deploy._path_to_label_list = str(session["path_to_label_list"])
        deploy.visualizeFlag = session.get("visualizeFlag", "robot") == "visualize"
        deploy.useCPU = session.get("useCPU", "CPU") == "CPU"
        deploy._intra_op_num_threads = int(session.get("intra_op_num_threads", 0))
        deploy._image_transport = str(session.get("image_transport", "raw"))
        deploy.publish_detection_segmentation = bool(
            session.get("publish_detection_segmentation", True)
        )
        deploy._confidence_threshold = float(session.get("confidence_threshold", 0.5))
        deploy._max_detections = int(session.get("max_detections", 100))
        deploy._input_image_topic = str(input_topic["input_image_topic"])
        deploy.usecase_mode = int(usecase["usecase_mode"])

        deploy.topic_button.blockSignals(True)
        deploy.topic_button.setEditText(deploy._input_image_topic)
        deploy.topic_button.blockSignals(False)

        transport_index = deploy.transport_combo.findText(deploy._image_transport)
        if transport_index >= 0:
            deploy.transport_combo.blockSignals(True)
            deploy.transport_combo.setCurrentIndex(transport_index)
            deploy.transport_combo.blockSignals(False)

        deploy.confidence_spinbox.blockSignals(True)
        deploy.confidence_spinbox.setValue(deploy._confidence_threshold)
        deploy.confidence_spinbox.blockSignals(False)
        deploy.max_detections_spinbox.blockSignals(True)
        deploy.max_detections_spinbox.setValue(deploy._max_detections)
        deploy.max_detections_spinbox.blockSignals(False)
        deploy._set_usecase_combo_to_mode(deploy.usecase_mode)
        deploy._update_fps_monitor_mode(deploy.usecase_mode)
        deploy.validateDeployInputs()

        model_controller = getattr(self.main_window, "_epd3_productization", None)
        if model_controller is not None and hasattr(model_controller, "schedule_inspection"):
            model_controller.schedule_inspection()
        model_controller = getattr(self.main_window, "_epd3_model_manager", None)
        if model_controller is not None and hasattr(model_controller, "schedule_inspection"):
            model_controller.schedule_inspection()

    def set_selected_known_good(self):
        path = self.dialog.selected_profile_path()
        if not path:
            self.dialog.profile_status.setText("Select a profile first.")
            return
        try:
            self.store.set_known_good(path)
        except ProfileError as exc:
            self._error("Unable to set known-good profile", exc)
            return
        self.dialog.profile_status.setText(
            f"Known-good profile set to {Path(path).name}."
        )
        self.refresh_profiles()

    def restore_known_good(self):
        try:
            path, _profile = self.store.restore_known_good()
        except ProfileError as exc:
            self._error("Unable to restore known-good profile", exc)
            return
        self._apply_profile_path(path)

    def import_profile(self):
        source, ok = QFileDialog.getOpenFileName(
            self.dialog,
            "Import EPD perception profile",
            str(Path.home()),
            "EPD Profile (*.json)",
        )
        if not ok or not source:
            return
        try:
            path = self.store.import_profile(source)
        except ProfileError as exc:
            self._error("Unable to import profile", exc)
            return
        self.dialog.profile_status.setText(f"Imported profile to {path}.")
        self.refresh_profiles()

    def export_profile(self):
        source = self.dialog.selected_profile_path()
        if not source:
            self.dialog.profile_status.setText("Select a profile first.")
            return
        suggested = Path.home() / Path(source).name
        destination, ok = QFileDialog.getSaveFileName(
            self.dialog,
            "Export EPD perception profile",
            str(suggested),
            "EPD Profile (*.json)",
        )
        if not ok or not destination:
            return
        try:
            exported = self.store.export_profile(source, destination)
        except ProfileError as exc:
            self._error("Unable to export profile", exc)
            return
        self.dialog.profile_status.setText(f"Exported profile to {exported}.")

    def delete_profile(self):
        path = self.dialog.selected_profile_path()
        if not path:
            self.dialog.profile_status.setText("Select a profile first.")
            return
        answer = QMessageBox.question(
            self.dialog,
            "Delete profile",
            f"Delete {Path(path).name}?",
        )
        if answer != QMessageBox.Yes:
            return
        self.store.delete(path)
        self.dialog.profile_status.setText("Profile deleted.")
        self.refresh_profiles()

    def choose_fixture(self):
        path, ok = QFileDialog.getOpenFileName(
            self.dialog,
            "Choose EPD replay fixture",
            str(self.package_root / "fixtures"),
            "Replay Fixture (*.json)",
        )
        if ok and path:
            self.dialog.fixture_path.setText(path)

    def choose_bag(self):
        path = QFileDialog.getExistingDirectory(
            self.dialog,
            "Choose rosbag2 directory",
            str(Path.home()),
        )
        if path:
            self.dialog.bag_path.setText(path)

    def run_fixture_replay(self):
        if self.process is not None and self.process.poll() is None:
            self.dialog.replay_state.setText("A replay process is already running.")
            return
        if self.deploy._job_controller.state in (JobState.STARTING, JobState.RUNNING):
            self.dialog.replay_state.setText(
                "Stop the normal Deploy session first; deterministic replay starts its own EPD node."
            )
            return
        fixture = Path(self.dialog.fixture_path.text().strip()).expanduser()
        if not fixture.is_file():
            self.dialog.replay_state.setText("Select a valid fixture JSON first.")
            return
        try:
            payload = json.loads(fixture.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.dialog.replay_state.setText(f"Invalid fixture: {exc}")
            return
        if payload.get("schema_version") != 1:
            self.dialog.replay_state.setText("Unsupported replay fixture schema version.")
            return
        self.summary_path.unlink(missing_ok=True)
        command = replay_command(
            fixture,
            self.dialog.fixture_mode.currentText(),
            self.summary_path,
        )
        self._start_process(command, "fixture")

    def inspect_bag(self):
        path = Path(self.dialog.bag_path.text().strip()).expanduser()
        if not path.is_dir():
            self.dialog.replay_state.setText("Select a rosbag2 directory first.")
            return
        if self.worker is not None and self.worker.is_alive():
            return
        self.dialog.replay_state.setText("INSPECTING BAG")
        self.worker = threading.Thread(
            target=self._inspect_bag_worker,
            args=(path,),
            daemon=True,
            name="EPD5BagInfo",
        )
        self.worker.start()

    def _inspect_bag_worker(self, path):
        try:
            result = subprocess.run(
                ["ros2", "bag", "info", str(path)],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.events.put(("bag_error", str(exc)))
            return
        if result.returncode != 0:
            self.events.put(("bag_error", result.stderr.strip() or "ros2 bag info failed"))
            return
        topics = parse_rosbag_topics(result.stdout)
        expected = self.deploy._input_image_topic.strip()
        self.events.put(("bag_info", (topics, expected, result.stdout)))

    def play_bag(self):
        if self.process is not None and self.process.poll() is None:
            self.dialog.replay_state.setText("A replay process is already running.")
            return
        if self.deploy._job_controller.state != JobState.RUNNING:
            self.dialog.replay_state.setText(
                "Start Deploy with the intended perception profile before playing a rosbag."
            )
            return
        path = Path(self.dialog.bag_path.text().strip()).expanduser()
        if not path.is_dir():
            self.dialog.replay_state.setText("Select a rosbag2 directory first.")
            return
        if shutil.which("ros2") is None:
            self.dialog.replay_state.setText("ros2 command is unavailable.")
            return
        self._start_process(rosbag_play_command(path), "rosbag")

    def _start_process(self, command, kind):
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.process = None
            self.dialog.replay_state.setText(f"Unable to start replay: {exc}")
            return
        self.process_kind = kind
        self.dialog.replay_details.clear()
        self.dialog.replay_state.setText("RUNNING " + kind.upper())
        threading.Thread(
            target=self._read_process_output,
            args=(self.process,),
            daemon=True,
            name="EPD5ReplayOutput",
        ).start()

    def _read_process_output(self, process):
        if process.stdout is not None:
            for line in process.stdout:
                self.events.put(("line", line.rstrip()))

    def stop_replay(self):
        process = self.process
        if process is None or process.poll() is not None:
            self.dialog.replay_state.setText("IDLE")
            return
        process.terminate()
        self.dialog.replay_state.setText("STOPPING")

    def _poll(self):
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                self.dialog.replay_details.append(
                    str(payload).replace("<", "&lt;").replace(">", "&gt;")
                )
            elif kind == "bag_error":
                self.dialog.replay_state.setText("BAG INSPECTION FAILED")
                self.dialog.replay_details.setPlainText(payload)
            elif kind == "bag_info":
                topics, expected, raw = payload
                self.dialog.replay_state.setText("BAG INSPECTED")
                lines = [f"Recorded topics ({len(topics)}):"] + topics
                if expected:
                    lines.extend(
                        [
                            "",
                            f"Active RGB topic: {expected}",
                            "Result: "
                            + (
                                "MATCH"
                                if expected in topics
                                else "MISSING — choose/apply the matching profile before playback"
                            ),
                        ]
                    )
                lines.extend(["", "Raw ros2 bag info:", raw])
                self.dialog.replay_details.setPlainText("\n".join(lines))

        if self.process is None:
            return
        return_code = self.process.poll()
        if return_code is None:
            return
        kind = self.process_kind or "replay"
        self.process = None
        self.process_kind = None
        if kind == "fixture" and self.summary_path.is_file():
            try:
                summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
                result = summary.get("result", "UNKNOWN")
                self.dialog.replay_state.setText(f"FIXTURE {result}")
                self.dialog.replay_details.append(
                    "\nAcceptance summary:\n"
                    + json.dumps(summary, indent=2, sort_keys=True)
                )
                return
            except (OSError, json.JSONDecodeError):
                pass
        state = "COMPLETE" if return_code == 0 else f"FAILED ({return_code})"
        self.dialog.replay_state.setText(f"{kind.upper()} {state}")

    def shutdown(self):
        self.stop_replay()
        if self.process is not None:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def _error(self, title, error):
        self.dialog.profile_status.setText(str(error))
        QMessageBox.warning(self.dialog, title, str(error))


def apply_epd5_productization(main_window):
    """Install EPD-5 once on a MainWindow instance."""
    existing = getattr(main_window, "_epd5_productization", None)
    if existing is not None:
        return existing
    controller = EPD5Controller(main_window)
    main_window._epd5_productization = controller
    return controller
