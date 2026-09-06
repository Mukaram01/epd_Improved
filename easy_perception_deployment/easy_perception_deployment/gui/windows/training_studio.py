"""EPD-4 Training Studio: observable, recoverable model training."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from types import MethodType

from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ITER_RE = re.compile(r"\biter(?:ation)?\s*:\s*(\d+)", re.IGNORECASE)
_LOSS_RE = re.compile(r"\bloss\s*:\s*([0-9.eE+\-]+)", re.IGNORECASE)
_LR_RE = re.compile(r"\blr\s*:\s*([0-9.eE+\-]+)", re.IGNORECASE)
_ETA_RE = re.compile(r"\beta\s*:\s*([^\s]+)", re.IGNORECASE)
_AP_RE = re.compile(r"Average Precision.*?=\s*([0-9.]+)", re.IGNORECASE)
_ITER_FROM_CHECKPOINT_RE = re.compile(r"model_(\d+)\.pth$")


def _read_coco(path):
    try:
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "annotations.json must contain a JSON object"
    return data, None


def dataset_summary(dataset_path, label_list=None):
    """Return file/annotation statistics without changing the dataset."""
    root = Path(dataset_path or "").expanduser()
    result = {
        "path": str(root),
        "valid": True,
        "splits": {},
        "warnings": [],
        "class_counts": {},
    }
    if not dataset_path or not root.is_dir():
        result["valid"] = False
        result["warnings"].append("Dataset directory is missing.")
        return result

    aggregate = Counter()
    for split_name in ("train_dataset", "val_dataset"):
        split = root / split_name
        entry = {
            "images_on_disk": 0,
            "coco_images": 0,
            "annotations": 0,
            "categories": 0,
        }
        if not split.is_dir():
            result["valid"] = False
            result["warnings"].append(f"Missing {split_name}/ directory.")
            result["splits"][split_name] = entry
            continue

        entry["images_on_disk"] = sum(
            1
            for item in split.iterdir()
            if item.is_file() and item.suffix.lower() in _IMAGE_EXTENSIONS
        )
        annotation_path = split / "annotations.json"
        data, error = _read_coco(annotation_path)
        if error:
            result["valid"] = False
            result["warnings"].append(
                f"{split_name}/annotations.json: {error}"
            )
            result["splits"][split_name] = entry
            continue

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        entry["coco_images"] = len(images) if isinstance(images, list) else 0
        entry["annotations"] = len(annotations) if isinstance(annotations, list) else 0
        entry["categories"] = len(categories) if isinstance(categories, list) else 0

        category_names = {}
        if isinstance(categories, list):
            for category in categories:
                if not isinstance(category, dict):
                    continue
                category_names[category.get("id")] = str(
                    category.get("name", category.get("id", "unknown"))
                )
        if isinstance(annotations, list):
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                category_id = annotation.get("category_id")
                aggregate[category_names.get(category_id, str(category_id))] += 1

        if entry["images_on_disk"] == 0:
            result["valid"] = False
            result["warnings"].append(f"{split_name} contains no images.")
        if entry["coco_images"] and entry["images_on_disk"] != entry["coco_images"]:
            result["warnings"].append(
                f"{split_name}: COCO lists {entry['coco_images']} images but "
                f"{entry['images_on_disk']} image files were found."
            )
        result["splits"][split_name] = entry

    result["class_counts"] = dict(sorted(aggregate.items()))
    nonzero = [value for value in aggregate.values() if value > 0]
    if len(nonzero) > 1 and max(nonzero) / min(nonzero) >= 10:
        result["warnings"].append(
            "Class imbalance is greater than 10:1; review sampling or collect more data."
        )

    labels = [str(item).strip() for item in (label_list or []) if str(item).strip()]
    if labels and aggregate:
        dataset_names = set(aggregate)
        ignored = {"__ignore__", "background", "__background__"}
        expected = {label for label in labels if label.lower() not in ignored}
        missing = sorted(expected - dataset_names)
        if missing:
            result["warnings"].append(
                "Labels with no annotations: " + ", ".join(missing[:12])
            )
    return result


def parse_training_line(line):
    """Parse common maskrcnn-benchmark progress output conservatively."""
    text = _ANSI_RE.sub("", str(line or "")).strip()
    event = {"raw": text}
    match = _ITER_RE.search(text)
    if match:
        event["iteration"] = int(match.group(1))
    match = _LOSS_RE.search(text)
    if match:
        try:
            event["loss"] = float(match.group(1))
        except ValueError:
            pass
    match = _LR_RE.search(text)
    if match:
        try:
            event["lr"] = float(match.group(1))
        except ValueError:
            pass
    match = _ETA_RE.search(text)
    if match:
        event["eta"] = match.group(1)
    match = _AP_RE.search(text)
    if match:
        try:
            event["validation_ap"] = float(match.group(1))
        except ValueError:
            pass
    return event


def training_guidance(history, max_iteration=0):
    """Explain trends without claiming validation evidence that is not present."""
    losses = [item["loss"] for item in history if "loss" in item]
    validation = [item["validation_ap"] for item in history if "validation_ap" in item]
    if len(losses) < 5:
        return (
            "Not enough training-loss samples yet. Watch the trend across many iterations, "
            "not a single noisy value."
        )

    first = sum(losses[: min(5, len(losses))]) / min(5, len(losses))
    last = sum(losses[-min(5, len(losses)) :]) / min(5, len(losses))
    iteration = max(
        (item.get("iteration", 0) for item in history),
        default=0,
    )
    progress = iteration / max_iteration if max_iteration else 0.0

    if last > first * 1.1 and progress > 0.25:
        return (
            "Training loss is rising relative to the early run. Check the learning rate, "
            "annotations and recent parameter changes before simply adding iterations."
        )
    if last > first * 0.9 and progress > 0.5:
        return (
            "Training loss has barely improved. This can indicate underfitting or a stalled "
            "optimization; inspect labels, model choice and learning-rate steps."
        )
    if validation:
        return (
            "Training loss is decreasing and validation AP is available. Prefer checkpoints "
            "with stronger validation AP rather than automatically choosing the last file."
        )
    return (
        "Training loss is trending down. Validation loss is not emitted by this legacy "
        "trainer, so overfitting cannot be diagnosed reliably from training loss alone. "
        "Use validation AP/checkpoint behaviour and real validation images before choosing "
        "the final model."
    )


def list_checkpoints(gui_dir, precision_level):
    """List current and archived checkpoints stored in the mounted trainer workspace."""
    trainer_name = "p3_trainer" if int(precision_level) == 3 else "p2_trainer"
    weights_root = Path(gui_dir) / trainer_name / "weights"
    records = []
    if not weights_root.is_dir():
        return records

    current_last = None
    last_file = weights_root / "custom" / "last_checkpoint"
    if last_file.is_file():
        try:
            current_last = last_file.read_text(encoding="utf-8").strip()
        except OSError:
            current_last = None

    directories = []
    current = weights_root / "custom"
    if current.is_dir():
        directories.append(("current", current))
    directories.extend(
        (path.name.replace("archived-on-", "archive "), path)
        for path in sorted(weights_root.glob("archived-on-*"), reverse=True)
        if path.is_dir()
    )

    for scope, directory in directories:
        for path in sorted(directory.glob("*.pth"), key=lambda item: item.stat().st_mtime, reverse=True):
            match = _ITER_FROM_CHECKPOINT_RE.search(path.name)
            iteration = int(match.group(1)) if match else None
            records.append(
                {
                    "scope": scope,
                    "path": str(path),
                    "name": path.name,
                    "iteration": iteration,
                    "mtime": path.stat().st_mtime,
                    "size": path.stat().st_size,
                    "latest": bool(
                        scope == "current"
                        and current_last
                        and Path(current_last).name == path.name
                    ),
                }
            )
    return records


def _copy_with_privilege_fallback(source, destination):
    try:
        shutil.copy2(source, destination)
        return
    except (OSError, PermissionError):
        result = subprocess.run(
            ["sudo", "cp", "--force", str(source), str(destination)],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Unable to copy checkpoint: {source}")


class TrainingStudioDialog(QDialog):
    """Operator-facing training telemetry, recovery and export surface."""

    def __init__(self, controller):
        super().__init__(controller.train)
        self.controller = controller
        self.setWindowTitle("EPD Training Studio")
        self.resize(980, 720)
        self.setMinimumSize(780, 600)

        outer = QVBoxLayout(self)
        title = QLabel("Training Studio", self)
        title.setObjectName("trainingStudioTitle")
        subtitle = QLabel(
            "Dataset truth, live training progress, checkpoints, resume and ONNX export.",
            self,
        )
        subtitle.setWordWrap(True)
        outer.addWidget(title)
        outer.addWidget(subtitle)

        self.tabs = QTabWidget(self)
        outer.addWidget(self.tabs, 1)
        self._build_dataset_tab()
        self._build_live_tab()
        self._build_checkpoint_tab()

        footer = QHBoxLayout()
        self.stop_button = QPushButton("Stop training", self)
        self.stop_button.clicked.connect(controller.stop_training)
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.close)
        footer.addWidget(self.stop_button)
        footer.addStretch(1)
        footer.addWidget(close_button)
        outer.addLayout(footer)
        self._apply_style()

    def _build_dataset_tab(self):
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)
        actions = QHBoxLayout()
        refresh = QPushButton("Refresh dataset summary", tab)
        refresh.clicked.connect(self.controller.refresh_dataset)
        actions.addWidget(refresh)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.dataset_text = QTextBrowser(tab)
        layout.addWidget(self.dataset_text, 1)
        self.tabs.addTab(tab, "Dataset")

    def _build_live_tab(self):
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)
        self.run_state = QLabel("IDLE", tab)
        self.run_state.setObjectName("trainingStudioState")
        layout.addWidget(self.run_state, 0, Qt.AlignLeft)

        self.progress = QProgressBar(tab)
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        grid = QGridLayout()
        self.iteration_value = QLabel("—", tab)
        self.loss_value = QLabel("—", tab)
        self.lr_value = QLabel("—", tab)
        self.eta_value = QLabel("—", tab)
        self.ap_value = QLabel("—", tab)
        fields = [
            ("Iteration", self.iteration_value),
            ("Training loss", self.loss_value),
            ("Learning rate", self.lr_value),
            ("ETA", self.eta_value),
            ("Validation AP", self.ap_value),
        ]
        for row, (name, value) in enumerate(fields):
            grid.addWidget(QLabel(name, tab), row, 0)
            grid.addWidget(value, row, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self.guidance = QLabel(tab)
        self.guidance.setWordWrap(True)
        self.guidance.setObjectName("trainingGuidance")
        layout.addWidget(self.guidance)

        self.log = QTextBrowser(tab)
        layout.addWidget(self.log, 1)
        self.tabs.addTab(tab, "Live training")

    def _build_checkpoint_tab(self):
        tab = QWidget(self.tabs)
        layout = QVBoxLayout(tab)
        intro = QLabel(
            "Resume preserves the selected checkpoint. Export selection is separate so you "
            "can compare checkpoints before choosing the model to deploy.",
            tab,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.checkpoints = QTableWidget(0, 5, tab)
        self.checkpoints.setHorizontalHeaderLabels(
            ["Scope", "Checkpoint", "Iteration", "Size", "Latest"]
        )
        self.checkpoints.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.checkpoints.setSelectionMode(QAbstractItemView.SingleSelection)
        self.checkpoints.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.checkpoints.verticalHeader().setVisible(False)
        self.checkpoints.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.checkpoints, 1)

        actions = QHBoxLayout()
        refresh = QPushButton("Refresh checkpoints", tab)
        refresh.clicked.connect(self.controller.refresh_checkpoints)
        resume = QPushButton("Resume selected", tab)
        resume.clicked.connect(self.controller.resume_selected)
        latest = QPushButton("Resume latest", tab)
        latest.clicked.connect(self.controller.resume_latest)
        fresh = QPushButton("Fresh run", tab)
        fresh.clicked.connect(self.controller.clear_resume)
        use_export = QPushButton("Use selected for export", tab)
        use_export.clicked.connect(self.controller.select_for_export)
        export_now = QPushButton("Export selected now", tab)
        export_now.clicked.connect(self.controller.export_selected_now)
        for button in (refresh, resume, latest, fresh, use_export, export_now):
            actions.addWidget(button)
        layout.addLayout(actions)

        self.checkpoint_status = QLabel("No checkpoint selected.", tab)
        self.checkpoint_status.setWordWrap(True)
        layout.addWidget(self.checkpoint_status)
        self.export_status = QLabel("ONNX export has not been validated in this session.", tab)
        self.export_status.setWordWrap(True)
        layout.addWidget(self.export_status)
        self.tabs.addTab(tab, "Checkpoints & export")

    def selected_checkpoint(self):
        row = self.checkpoints.currentRow()
        if row < 0:
            return None
        item = self.checkpoints.item(row, 1)
        return item.data(Qt.UserRole) if item is not None else None

    def set_dataset_summary(self, summary):
        parts = [f"<b>Dataset:</b> {summary.get('path', '—')}"]
        for name, split in summary.get("splits", {}).items():
            parts.append(
                f"<p><b>{name}</b><br>"
                f"Images on disk: {split.get('images_on_disk', 0)}<br>"
                f"COCO images: {split.get('coco_images', 0)}<br>"
                f"Annotations: {split.get('annotations', 0)}<br>"
                f"Categories: {split.get('categories', 0)}</p>"
            )
        counts = summary.get("class_counts", {})
        if counts:
            parts.append("<b>Annotation counts by class</b><br>" + "<br>".join(
                f"{name}: {count}" for name, count in counts.items()
            ))
        warnings = summary.get("warnings", [])
        if warnings:
            parts.append("<p><b>Review</b><br>" + "<br>".join(warnings) + "</p>")
        else:
            parts.append("<p><b>Dataset summary:</b> no structural warnings found.</p>")
        self.dataset_text.setHtml("\n".join(parts))

    def set_checkpoints(self, records):
        self.checkpoints.setRowCount(len(records))
        for row, record in enumerate(records):
            values = [
                record["scope"],
                record["name"],
                "—" if record["iteration"] is None else str(record["iteration"]),
                f"{record['size'] / (1024 * 1024):.1f} MB",
                "yes" if record["latest"] else "",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1:
                    item.setData(Qt.UserRole, record["path"])
                self.checkpoints.setItem(row, column, item)
        self.checkpoints.resizeColumnsToContents()

    def append_log(self, line):
        if line:
            self.log.append(line.replace("<", "&lt;").replace(">", "&gt;"))

    def _apply_style(self):
        self.setStyleSheet(
            """
            QDialog { background: #15191f; color: #e8edf2; }
            QLabel#trainingStudioTitle { font-size: 24px; font-weight: 700; }
            QLabel#trainingStudioState {
                background: #29313a; border-radius: 6px; padding: 6px 10px;
                font-weight: 700;
            }
            QLabel#trainingGuidance {
                background: #202731; border: 1px solid #394452;
                border-radius: 6px; padding: 10px;
            }
            QTextBrowser, QTableWidget {
                background: #11161b; border: 1px solid #303944; border-radius: 6px;
            }
            QPushButton { min-height: 32px; padding: 0 10px; }
            """
        )


class TrainingStudioController(QObject):
    """Connect the legacy trainer to observable EPD-4 behaviour."""

    def __init__(self, main_window):
        super().__init__(main_window.train_window)
        self.main_window = main_window
        self.train = main_window.train_window
        self.gui_dir = Path(self.train._GUI_DIR)
        self.package_root = Path(self.train._PACKAGE_ROOT)
        self.events = queue.Queue()
        self.history = []
        self.resume_checkpoint = None
        self.export_checkpoint = None
        self.current_iteration = 0
        self.current_loss = None
        self.current_lr = None
        self.current_eta = None
        self.current_ap = None
        self.process = None
        self._export_thread = None
        self._last_onnx_mtime = 0.0
        self.dialog = TrainingStudioDialog(self)
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._install_button()
        self._patch_training_entrypoint()
        self.refresh_dataset()
        self.refresh_checkpoints()

    def _install_button(self):
        button = QPushButton("Training Studio", self.train)
        button.setToolTip(
            "Open live training metrics, dataset statistics, checkpoints, resume and export."
        )
        button.clicked.connect(self.show)
        self.train.training_studio_button = button
        controller = getattr(self.main_window, "_train_ui_controller", None)
        header_badge = getattr(controller, "header_badge", None)
        header = header_badge.parentWidget() if header_badge is not None else None
        if header is not None and header.layout() is not None:
            index = max(0, header.layout().count() - 1)
            header.layout().insertWidget(index, button, 0, Qt.AlignTop)
        else:
            self.train.layout().addWidget(button)

    def _patch_training_entrypoint(self):
        controller = self

        def epd4_start_training(_window):
            return controller._start_training()

        self.train.startTraining = MethodType(epd4_start_training, self.train)

    def show(self):
        self.refresh_dataset()
        self.refresh_checkpoints()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def refresh_dataset(self):
        summary = dataset_summary(
            self.train._path_to_dataset,
            self.train._label_list,
        )
        self.dialog.set_dataset_summary(summary)
        return summary

    def refresh_checkpoints(self):
        records = list_checkpoints(self.gui_dir, self.train._precision_level)
        self.dialog.set_checkpoints(records)
        return records

    def resume_selected(self):
        selected = self.dialog.selected_checkpoint()
        if not selected:
            self.dialog.checkpoint_status.setText("Select a checkpoint first.")
            return
        self.resume_checkpoint = selected
        self.dialog.checkpoint_status.setText(
            f"Next Train action will continue from: {Path(selected).name}"
        )

    def resume_latest(self):
        records = self.refresh_checkpoints()
        selected = next((item for item in records if item["latest"]), None)
        if selected is None:
            selected = next((item for item in records if item["scope"] == "current"), None)
        if selected is None:
            self.dialog.checkpoint_status.setText("No current checkpoint is available to resume.")
            return
        self.resume_checkpoint = selected["path"]
        self.dialog.checkpoint_status.setText(
            f"Next Train action will continue from: {selected['name']}"
        )

    def clear_resume(self):
        self.resume_checkpoint = None
        self.dialog.checkpoint_status.setText(
            "Next Train action will start a fresh run and archive current weights."
        )

    def select_for_export(self):
        selected = self.dialog.selected_checkpoint()
        if not selected:
            self.dialog.checkpoint_status.setText("Select a checkpoint first.")
            return
        self.export_checkpoint = selected
        self.dialog.checkpoint_status.setText(
            f"Checkpoint selected for ONNX export: {Path(selected).name}"
        )

    def _trainer_values(self):
        return (
            self.train._path_to_dataset,
            self.train.model_name,
            self.train._label_list,
            self.train.max_iteration,
            self.train.checkpoint_period,
            self.train.test_period,
            self.train.steps,
        )

    def _new_trainer(self):
        if self.train._precision_level == 2:
            from trainer.P2Trainer import P2Trainer

            return P2Trainer(*self._trainer_values())
        if self.train._precision_level == 3:
            from trainer.P3Trainer import P3Trainer

            return P3Trainer(*self._trainer_values())
        raise RuntimeError("Precision Level 1 training is deprecated.")

    def _start_training(self):
        trainer = self._new_trainer()
        if not getattr(trainer, "isGPUAvailableFlag", False):
            raise RuntimeError(
                "Training requires the GPU/CUDA trainer environment; nvidia-smi or nvcc failed."
            )

        self.history = []
        self.current_iteration = 0
        self.current_loss = None
        self.current_lr = None
        self.current_eta = None
        self.current_ap = None
        self.dialog.log.clear()
        self._last_onnx_mtime = self._newest_onnx_mtime()

        trainer.copyTrainingFiles = MethodType(
            lambda instance: self._copy_training_files(instance),
            trainer,
        )
        trainer.runTraining = MethodType(
            lambda instance: self._run_training_process(instance),
            trainer,
        )
        trainer.runExporter = MethodType(
            lambda instance: self._run_exporter_process(instance),
            trainer,
        )

        self.events.put(("phase", "TRAINING"))
        trainer.train(False)

        selected = self.export_checkpoint
        if selected:
            _copy_with_privilege_fallback(selected, self.gui_dir / "trained.pth")
        trainer.export(False)
        self._validate_latest_onnx()
        self.resume_checkpoint = None
        self.events.put(("phase", "COMPLETE"))
        self.refresh_checkpoints()

    def _paths_for_precision(self):
        if self.train._precision_level == 3:
            return {
                "container": "epd_p3_trainer",
                "trainer_root": "/home/user/p3_trainer",
                "host_trainer": self.gui_dir / "p3_trainer",
                "config": "configs/custom/maskrcnn_training.yaml",
                "host_config": self.gui_dir / "trainer/training_files/maskrcnn_training.yaml",
            }
        return {
            "container": "epd_p2_trainer",
            "trainer_root": "/home/user/p2_trainer",
            "host_trainer": self.gui_dir / "p2_trainer",
            "config": "configs/custom/fasterrcnn_training.yaml",
            "host_config": self.gui_dir / "trainer/training_files/fasterrcnn_training.yaml",
        }

    def _stage_dataset(self):
        source = Path(self.train._path_to_dataset)
        staging = self.gui_dir / "trainer/training_files/custom_dataset"
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError:
                subprocess.run(["sudo", "rm", "-rf", str(staging)], check=True)
        shutil.copytree(source, staging)
        return staging

    def _copy_training_files(self, trainer):
        paths = self._paths_for_precision()
        self._stage_dataset()
        subprocess.run(
            ["sudo", "docker", "start", paths["container"]],
            check=False,
            stdout=subprocess.DEVNULL,
        )
        preserve = bool(self.resume_checkpoint)
        archive = ""
        if not preserve:
            archive = (
                "if [ -d weights/custom ]; then "
                "stamp=$(date +%Y-%m-%d-%H-%M-%S); "
                "mv weights/custom weights/archived-on-$stamp; "
                "mkdir -p weights/custom; fi; "
            )
        command = (
            f"cd {paths['trainer_root']} && "
            + archive
            + "rm -rf datasets/custom_dataset && "
            + "cp -r /home/user/trainer/training_files/custom_dataset datasets/custom_dataset && "
            + f"cp --force /home/user/{paths['host_config'].relative_to(self.gui_dir)} "
            + f"{paths['config']}"
        )
        result = subprocess.run(
            ["sudo", "docker", "exec", paths["container"], "sh", "-lc", command],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Unable to copy dataset/config into the training container.")

    def _container_checkpoint(self, host_path):
        if not host_path:
            return None
        path = Path(host_path).resolve()
        try:
            relative = path.relative_to(self.gui_dir.resolve())
        except ValueError:
            raise RuntimeError("Resume checkpoint must be inside the EPD trainer workspace.")
        return "/home/user/" + relative.as_posix()

    def _run_training_process(self, trainer):
        paths = self._paths_for_precision()
        command = [
            "sudo",
            "docker",
            "exec",
            "-i",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-w",
            paths["trainer_root"],
            paths["container"],
            "python",
            "tools/train_net.py",
            "--config-file",
            paths["config"],
        ]
        resume_path = self._container_checkpoint(self.resume_checkpoint)
        if resume_path:
            command.extend(["MODEL.WEIGHT", resume_path])
            self.events.put(("line", f"Resuming from {resume_path}"))
        self._stream_process(command, "training")

        final = paths["host_trainer"] / "weights/custom/model_final.pth"
        if not final.is_file():
            raise RuntimeError("Training completed without weights/custom/model_final.pth.")
        _copy_with_privilege_fallback(final, self.gui_dir / "trained.pth")

    def _run_exporter_process(self, trainer):
        mask = "true" if self.train._precision_level == 3 else "false"
        command = [
            "bash",
            "trainer/exporter_files/scripts/run_exporter.bash",
            mask,
        ]
        self.events.put(("phase", "EXPORTING"))
        self._stream_process(command, "export")

        output = self.gui_dir / "output.onnx"
        if not output.is_file():
            raise RuntimeError("Exporter finished without output.onnx.")

        timestamp = time.strftime("%d-%m-%Y-%H-%M-%S")
        model_name = self.train.model_name or "trained-model"
        destination = self.package_root / "data/model" / f"{model_name}-{timestamp}.onnx"
        shutil.copy2(output, destination)
        self.events.put(("line", f"ONNX saved to {destination}"))

    def _stream_process(self, command, phase):
        self.events.put(("phase", phase.upper()))
        process = subprocess.Popen(
            command,
            cwd=str(self.gui_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.process = process
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    self.events.put(("line", line.rstrip()))
            return_code = process.wait()
        finally:
            self.process = None
        if return_code != 0:
            raise RuntimeError(f"{phase.capitalize()} process exited with code {return_code}.")

    def stop_training(self):
        process = self.process
        if process is None or process.poll() is not None:
            self.dialog.run_state.setText("No active training/export process.")
            return
        process.terminate()
        paths = self._paths_for_precision()
        subprocess.run(
            [
                "sudo",
                "docker",
                "exec",
                paths["container"],
                "pkill",
                "-INT",
                "-f",
                "tools/train_net.py",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.events.put(("phase", "STOPPING"))

    def export_selected_now(self):
        selected = self.dialog.selected_checkpoint() or self.export_checkpoint
        if not selected:
            self.dialog.export_status.setText("Select a checkpoint to export first.")
            return
        if self._export_thread is not None and self._export_thread.is_alive():
            self.dialog.export_status.setText("An export is already running.")
            return
        self.export_checkpoint = selected
        self._export_thread = threading.Thread(
            target=self._export_only_worker,
            name="epd4Export",
            daemon=True,
        )
        self._export_thread.start()

    def _export_only_worker(self):
        try:
            trainer = self._new_trainer()
            if not getattr(trainer, "isGPUAvailableFlag", False):
                raise RuntimeError("GPU/CUDA exporter environment is not ready.")
            _copy_with_privilege_fallback(
                self.export_checkpoint,
                self.gui_dir / "trained.pth",
            )
            trainer.runExporter = MethodType(
                lambda instance: self._run_exporter_process(instance),
                trainer,
            )
            self._last_onnx_mtime = self._newest_onnx_mtime()
            trainer.export(False)
            self._validate_latest_onnx()
            self.events.put(("phase", "EXPORT COMPLETE"))
        except Exception as exc:
            self.events.put(("export_error", str(exc)))

    def _newest_onnx_mtime(self):
        files = list((self.package_root / "data/model").glob("*.onnx"))
        return max((path.stat().st_mtime for path in files), default=0.0)

    def _validate_latest_onnx(self):
        files = sorted(
            (self.package_root / "data/model").glob("*.onnx"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not files or files[0].stat().st_mtime < self._last_onnx_mtime:
            self.events.put(("onnx", "No newly exported ONNX model was found."))
            return
        model = files[0]
        try:
            from windows.model_manager import inspect_deployment_model

            mode = 3 if self.train._precision_level == 3 else 1
            result = inspect_deployment_model(
                str(model),
                self.train._path_to_label_list,
                mode,
                self.package_root,
            )
            summary = result.get("summary", "Inspection completed.")
            status = result.get("status", "unknown").upper()
            self.events.put(("onnx", f"{status}: {model.name} — {summary}"))
        except Exception as exc:
            self.events.put(("onnx", f"ONNX validation could not complete: {exc}"))

    def _tick(self):
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                self._consume_line(payload)
            elif kind == "phase":
                self.dialog.run_state.setText(payload)
            elif kind == "onnx":
                self.dialog.export_status.setText(payload)
            elif kind == "export_error":
                self.dialog.export_status.setText("Export failed: " + payload)

        max_iter = max(1, int(self.train.max_iteration))
        self.dialog.progress.setMaximum(max_iter)
        self.dialog.progress.setValue(min(self.current_iteration, max_iter))
        self.dialog.iteration_value.setText(
            f"{self.current_iteration} / {max_iter}" if self.current_iteration else f"— / {max_iter}"
        )
        self.dialog.loss_value.setText(
            "—" if self.current_loss is None else f"{self.current_loss:.5g}"
        )
        self.dialog.lr_value.setText(
            "—" if self.current_lr is None else f"{self.current_lr:.5g}"
        )
        self.dialog.eta_value.setText(self.current_eta or "—")
        self.dialog.ap_value.setText(
            "—" if self.current_ap is None else f"{self.current_ap:.4f}"
        )
        self.dialog.guidance.setText(training_guidance(self.history, max_iter))
        state = getattr(self.train._job_controller.state, "name", str(self.train._job_controller.state))
        if self.process is None and self.dialog.run_state.text() in {"IDLE", "TRAINING"}:
            self.dialog.run_state.setText(state)

    def _consume_line(self, line):
        event = parse_training_line(line)
        if "iteration" in event:
            self.current_iteration = event["iteration"]
        if "loss" in event:
            self.current_loss = event["loss"]
        if "lr" in event:
            self.current_lr = event["lr"]
        if "eta" in event:
            self.current_eta = event["eta"]
        if "validation_ap" in event:
            self.current_ap = event["validation_ap"]
            event.setdefault("iteration", self.current_iteration)
        if any(key in event for key in ("iteration", "loss", "validation_ap")):
            self.history.append(event)
        self.dialog.append_log(line)


def apply_training_studio(main_window):
    """Install EPD-4 on one MainWindow instance."""
    existing = getattr(main_window, "_epd4_training_studio", None)
    if existing is not None:
        return existing
    controller = TrainingStudioController(main_window)
    main_window._epd4_training_studio = controller
    return controller
