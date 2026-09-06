"""EPD-8 execution-backend discovery, selection, and benchmarking UI."""

from __future__ import annotations

import json
import os
import platform
import queue
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import MethodType

from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
)

from windows.job_controller import JobState

BACKENDS = ("auto", "cpu", "cuda", "tensorrt")
BACKEND_LABELS = {
    "auto": "Auto (CUDA when verified, otherwise CPU)",
    "cpu": "CPU",
    "cuda": "NVIDIA CUDA",
    "tensorrt": "NVIDIA TensorRT",
}


def normalize_backend(value, legacy_use_cpu=True):
    value = str(value or "").strip().lower()
    aliases = {
        "gpu": "cuda",
        "nvidia": "cuda",
        "trt": "tensorrt",
        "default": "auto",
    }
    value = aliases.get(value, value)
    if not value:
        return "cpu" if legacy_use_cpu else "cuda"
    return value if value in BACKENDS else "auto"


def _run(command, timeout=4.0, env=None):
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _docker_command():
    override = os.getenv("EPD_DOCKER_CMD", "").strip()
    if override:
        command = shlex.split(override)
        return command if command else ["docker"]
    return ["docker"]


def _docker_image_exists(command, image):
    if not image:
        return False
    result = _run(command + ["image", "inspect", image], timeout=3.0)
    return result is not None and result.returncode == 0


def _parse_last_json(stdout):
    for line in reversed(str(stdout or "").splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _compiled_capabilities():
    if shutil.which("ros2") is None:
        return None, "ros2 command unavailable"
    result = _run(
        ["ros2", "run", "easy_perception_deployment", "epd_backend_probe"],
        timeout=5.0,
    )
    if result is None:
        return None, "compiled backend probe timed out"
    payload = _parse_last_json(result.stdout)
    if result.returncode != 0 or payload is None:
        detail = result.stderr.strip() or "probe executable is not built/installed"
        return None, detail
    return payload, ""


def _jetson_info():
    release = Path("/etc/nv_tegra_release")
    model_path = Path("/proc/device-tree/model")
    model = ""
    if model_path.is_file():
        try:
            model = model_path.read_text(
                encoding="utf-8", errors="ignore").replace("\x00", "").strip()
        except OSError:
            model = ""
    is_jetson = release.is_file() or (
        platform.machine().lower() in ("aarch64", "arm64")
        and "nvidia" in model.lower()
    )
    return is_jetson, model


def probe_environment():
    """Return host/container/build evidence without claiming more than measured."""
    docker = _docker_command()
    docker_version = _run(docker + ["--version"], timeout=3.0)
    docker_ok = docker_version is not None and docker_version.returncode == 0

    runtimes = None
    if docker_ok:
        runtimes = _run(
            docker + ["info", "--format", "{{json .Runtimes}}"],
            timeout=4.0,
        )
    nvidia_runtime = bool(
        runtimes is not None
        and runtimes.returncode == 0
        and "nvidia" in runtimes.stdout.lower()
    )

    smi = _run(["nvidia-smi", "-L"], timeout=3.0) if shutil.which(
        "nvidia-smi") else None
    nvidia_smi = smi is not None and smi.returncode == 0
    gpu_text = ""
    if nvidia_smi:
        gpu_text = smi.stdout.strip().splitlines()[0] if smi.stdout.strip() else "Detected"

    jetson, jetson_model = _jetson_info()
    nvidia_host = nvidia_smi or jetson
    nvidia_container = nvidia_runtime or nvidia_smi or jetson

    cpu_image = os.getenv(
        "EPD_CPU_IMAGE", "cardboardcode/epd-humble-base:CPU")
    gpu_image = os.getenv(
        "EPD_GPU_IMAGE", "cardboardcode/epd-humble-base:GPU")
    tensorrt_image = os.getenv("EPD_TENSORRT_IMAGE", "").strip()

    images = {
        "cpu": {
            "name": cpu_image,
            "present": docker_ok and _docker_image_exists(docker, cpu_image),
        },
        "cuda": {
            "name": gpu_image,
            "present": docker_ok and _docker_image_exists(docker, gpu_image),
        },
        "tensorrt": {
            "name": tensorrt_image,
            "present": bool(
                tensorrt_image
                and docker_ok
                and _docker_image_exists(docker, tensorrt_image)
            ),
        },
    }

    compiled, compiled_error = _compiled_capabilities()
    compiled_cpu = True if compiled is None else bool(compiled.get("cpu", True))
    compiled_cuda = None if compiled is None else bool(compiled.get("cuda", False))
    compiled_trt = None if compiled is None else bool(compiled.get("tensorrt", False))

    cpu_ready = docker_ok and images["cpu"]["present"] and compiled_cpu
    cuda_ready = (
        docker_ok
        and nvidia_host
        and nvidia_container
        and images["cuda"]["present"]
        and compiled_cuda is not False
    )
    trt_ready = (
        docker_ok
        and nvidia_host
        and nvidia_container
        and images["tensorrt"]["present"]
        and compiled_trt is True
    )

    if cuda_ready:
        recommended = "cuda"
    else:
        recommended = "cpu"

    return {
        "architecture": platform.machine(),
        "jetson": jetson,
        "jetson_model": jetson_model,
        "docker_ok": docker_ok,
        "docker_version": (
            docker_version.stdout.strip() if docker_version is not None else ""
        ),
        "nvidia_host": nvidia_host,
        "nvidia_smi": nvidia_smi,
        "gpu_text": gpu_text,
        "nvidia_container_runtime": nvidia_container,
        "compiled": compiled,
        "compiled_error": compiled_error,
        "images": images,
        "ready": {
            "cpu": cpu_ready,
            "cuda": cuda_ready,
            "tensorrt": trt_ready,
        },
        "recommended": recommended,
    }


def backend_status(probe, backend):
    if backend == "auto":
        resolved = probe.get("recommended", "cpu")
        ready = probe.get("ready", {}).get(resolved, False)
        return (
            "READY" if ready else "CHECK",
            f"AUTO currently prefers {resolved.upper()} based on measured host/runtime evidence.",
        )
    ready = probe.get("ready", {}).get(backend, False)
    if ready:
        return "READY", f"{backend.upper()} prerequisites are currently verified."
    if backend == "tensorrt":
        if not probe.get("images", {}).get("tensorrt", {}).get("name"):
            return (
                "BLOCKED",
                "TensorRT requires EPD_TENSORRT_IMAGE plus an EPD build compiled with the TensorRT provider.",
            )
        if probe.get("compiled") is None:
            return (
                "CHECK",
                "TensorRT image is configured, but compiled provider capability could not be verified yet.",
            )
    if backend == "cuda" and probe.get("compiled") is None:
        return (
            "CHECK",
            "CUDA host/container evidence is incomplete or the compiled capability probe is unavailable.",
        )
    return "BLOCKED", f"{backend.upper()} prerequisites are not currently verified."


def _read_json(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=4) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class PerformanceBackendsDialog(QDialog):
    def __init__(self, controller):
        super().__init__(controller.deploy)
        self.controller = controller
        self.setWindowTitle("EPD Performance Backends")
        self.resize(980, 720)
        self.setMinimumSize(800, 600)

        outer = QVBoxLayout(self)
        title = QLabel("Performance Backends", self)
        title.setObjectName("backendTitle")
        subtitle = QLabel(
            "Choose CPU, CUDA or an explicitly provisioned TensorRT runtime. "
            "AUTO uses CUDA only when the host/container path is verified; otherwise it falls back to CPU.",
            self,
        )
        subtitle.setWordWrap(True)
        outer.addWidget(title)
        outer.addWidget(subtitle)

        selection = QGridLayout()
        selection.addWidget(QLabel("Execution backend", self), 0, 0)
        self.backend_combo = QComboBox(self)
        for backend in BACKENDS:
            self.backend_combo.addItem(BACKEND_LABELS[backend], backend)
        selection.addWidget(self.backend_combo, 0, 1)
        selection.addWidget(QLabel("GPU index", self), 1, 0)
        self.gpu_index = QSpinBox(self)
        self.gpu_index.setRange(0, 31)
        selection.addWidget(self.gpu_index, 1, 1)
        self.apply_button = QPushButton("Apply backend", self)
        self.apply_button.clicked.connect(controller.apply_selection)
        selection.addWidget(self.apply_button, 0, 2, 2, 1)
        outer.addLayout(selection)

        self.selection_status = QLabel("", self)
        self.selection_status.setWordWrap(True)
        self.selection_status.setObjectName("backendStatus")
        outer.addWidget(self.selection_status)

        probe_actions = QHBoxLayout()
        probe_label = QLabel("Measured environment", self)
        probe_label.setObjectName("backendSection")
        probe_actions.addWidget(probe_label)
        probe_actions.addStretch(1)
        probe_button = QPushButton("Probe again", self)
        probe_button.clicked.connect(controller.refresh_probe)
        probe_actions.addWidget(probe_button)
        outer.addLayout(probe_actions)

        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Check", "State", "Detail"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        outer.addWidget(self.table, 1)

        benchmark_title = QLabel("Deterministic CPU vs accelerated benchmark", self)
        benchmark_title.setObjectName("backendSection")
        outer.addWidget(benchmark_title)
        benchmark_note = QLabel(
            "Benchmarking reuses the existing P8 replay acceptance fixture. "
            "A faster backend is not accepted merely because it is faster: replay must PASS, "
            "and accelerated semantic summaries are compared with the CPU baseline.",
            self,
        )
        benchmark_note.setWordWrap(True)
        outer.addWidget(benchmark_note)

        benchmark_row = QHBoxLayout()
        self.benchmark_combo = QComboBox(self)
        self.benchmark_combo.addItem("CPU only", "cpu")
        self.benchmark_combo.addItem("CPU vs CUDA", "cpu,cuda")
        self.benchmark_combo.addItem("CPU vs TensorRT", "cpu,tensorrt")
        self.benchmark_combo.addItem(
            "CPU vs CUDA vs TensorRT", "cpu,cuda,tensorrt")
        benchmark_row.addWidget(self.benchmark_combo)
        self.benchmark_button = QPushButton("Run benchmark", self)
        self.benchmark_button.clicked.connect(controller.run_benchmark)
        benchmark_row.addWidget(self.benchmark_button)
        benchmark_row.addStretch(1)
        outer.addLayout(benchmark_row)

        self.benchmark_output = QTextBrowser(self)
        self.benchmark_output.setMinimumHeight(150)
        outer.addWidget(self.benchmark_output)

        footer = QHBoxLayout()
        footer.addStretch(1)
        close = QPushButton("Close", self)
        close.clicked.connect(self.hide)
        footer.addWidget(close)
        outer.addLayout(footer)
        self._style()

    def set_backend(self, backend, gpu_index):
        index = self.backend_combo.findData(backend)
        if index >= 0:
            self.backend_combo.setCurrentIndex(index)
        self.gpu_index.setValue(int(gpu_index))

    def show_probe(self, probe, backend):
        state, detail = backend_status(probe, backend)
        self.selection_status.setText(
            f"Selected: {backend.upper()} • {state} — {detail}"
        )
        rows = []
        rows.append((
            "Architecture",
            "JETSON" if probe["jetson"] else probe["architecture"],
            probe["jetson_model"] or probe["architecture"],
        ))
        rows.append((
            "Docker",
            "READY" if probe["docker_ok"] else "MISSING",
            probe["docker_version"] or "Docker CLI/daemon unavailable",
        ))
        rows.append((
            "NVIDIA host",
            "READY" if probe["nvidia_host"] else "NOT DETECTED",
            probe["gpu_text"] or (
                "Jetson platform" if probe["jetson"] else "No NVIDIA GPU evidence"
            ),
        ))
        rows.append((
            "NVIDIA container runtime",
            "READY" if probe["nvidia_container_runtime"] else "NOT DETECTED",
            "Required for CUDA/TensorRT Docker deployment",
        ))
        compiled = probe.get("compiled")
        if compiled is None:
            compiled_detail = probe.get("compiled_error") or "Unavailable"
            rows.append(("Compiled providers", "UNKNOWN", compiled_detail))
        else:
            providers = ["CPU"]
            if compiled.get("cuda"):
                providers.append("CUDA")
            if compiled.get("tensorrt"):
                providers.append("TensorRT")
            rows.append(("Compiled providers", "READY", ", ".join(providers)))

        for backend_name in ("cpu", "cuda", "tensorrt"):
            image = probe["images"][backend_name]
            if backend_name == "tensorrt" and not image["name"]:
                image_state = "NOT CONFIGURED"
                image_detail = "Set EPD_TENSORRT_IMAGE after building a TensorRT-capable runtime image"
            else:
                image_state = "READY" if image["present"] else "MISSING"
                image_detail = image["name"] or "—"
            rows.append((f"{backend_name.upper()} image", image_state, image_detail))

        rows.append((
            "AUTO recommendation",
            probe["recommended"].upper(),
            "AUTO never chooses TensorRT implicitly; TensorRT must be selected and benchmarked explicitly.",
        ))

        self.table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()

    def _style(self):
        self.setStyleSheet(
            """
            QDialog { background: #15191f; color: #e8edf2; }
            QLabel#backendTitle { font-size: 24px; font-weight: 700; }
            QLabel#backendSection { font-size: 15px; font-weight: 700; margin-top: 8px; }
            QLabel#backendStatus {
                background: #202832; border: 1px solid #354155;
                border-radius: 7px; padding: 8px 10px;
            }
            QTableWidget, QTextBrowser, QComboBox, QSpinBox {
                background: #11161b; border: 1px solid #303944; border-radius: 6px;
            }
            QPushButton { min-height: 32px; padding: 0 10px; }
            """
        )


class PerformanceBackendsController(QObject):
    """Add EPD-8 backend truth without changing scene/task/motion ownership."""

    def __init__(self, main_window):
        super().__init__(main_window.deploy_window)
        self.main_window = main_window
        self.deploy = main_window.deploy_window
        self.package_root = Path(self.deploy._PACKAGE_ROOT)
        self.config_path = self.package_root / "config" / "session_config.json"
        self.backend = "auto"
        self.gpu_index = 0
        self.probe = {}
        self.events = queue.Queue()
        self._benchmark_thread = None
        self._original_update_session_config = self.deploy.updateSessionConfig
        self._load_selection()
        self.dialog = PerformanceBackendsDialog(self)
        self.dialog.set_backend(self.backend, self.gpu_index)
        self._install_update_hook()
        self._install_profile_hook()
        self._install_controls()
        self.refresh_probe()

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(200)
        self.poll_timer.timeout.connect(self._poll_events)
        self.poll_timer.start()

        self.label_timer = QTimer(self)
        self.label_timer.setInterval(180)
        self.label_timer.timeout.connect(self._sync_compact_label)
        self.label_timer.start()
        self._sync_compact_label()

    def _load_selection(self):
        config = _read_json(self.config_path)
        legacy_cpu = str(config.get("useCPU", "CPU")) == "CPU"
        self.backend = normalize_backend(
            config.get("execution_backend"), legacy_cpu)
        try:
            self.gpu_index = max(
                0, int(config.get("execution_backend_gpu_index", 0)))
        except (TypeError, ValueError):
            self.gpu_index = 0

    def _install_update_hook(self):
        controller = self
        original = self._original_update_session_config

        def update_and_preserve_backend(_deploy, *args, **kwargs):
            result = original(*args, **kwargs)
            controller._persist_backend_fields()
            return result

        self.deploy.updateSessionConfig = MethodType(
            update_and_preserve_backend, self.deploy)

    def _install_profile_hook(self):
        epd5 = getattr(self.main_window, "_epd5_productization", None)
        if epd5 is None or getattr(epd5, "_epd8_backend_hook", False):
            return
        epd5._epd8_backend_hook = True
        original = epd5._sync_deploy
        controller = self

        def sync_profile_and_backend(profile_controller, profile):
            session = (profile.get("epd") or {}).get("session_config") or {}
            legacy_cpu = str(session.get("useCPU", "CPU")) == "CPU"
            controller.backend = normalize_backend(
                session.get("execution_backend"), legacy_cpu)
            try:
                controller.gpu_index = max(
                    0, int(session.get("execution_backend_gpu_index", 0)))
            except (TypeError, ValueError):
                controller.gpu_index = 0
            result = original(profile)
            controller.dialog.set_backend(
                controller.backend, controller.gpu_index)
            controller._persist_backend_fields()
            controller._sync_compact_label()
            return result

        epd5._sync_deploy = MethodType(sync_profile_and_backend, epd5)

    def _install_controls(self):
        # Replace the old blind CPU/GPU toggle with an explicit manager. The
        # legacy boolean is still maintained for old profiles/scripts.
        try:
            self.deploy.docker_button.clicked.disconnect()
        except RuntimeError:
            pass
        self.deploy.docker_button.clicked.connect(self.show)
        self.deploy.docker_button.setToolTip(
            "Open Performance Backends. Select AUTO, CPU, CUDA or a verified TensorRT runtime."
        )

        self.header_button = QPushButton("Performance", self.deploy)
        self.header_button.setToolTip(
            "Inspect CPU/GPU/TensorRT/Jetson readiness and benchmark backends."
        )
        self.header_button.clicked.connect(self.show)
        ui = getattr(self.main_window, "_deploy_ui_controller", None)
        badge = getattr(ui, "header_badge", None)
        header = badge.parentWidget() if badge is not None else None
        if header is not None and header.layout() is not None:
            index = max(0, header.layout().count() - 1)
            header.layout().insertWidget(
                index, self.header_button, 0, Qt.AlignTop)

    def show(self):
        self.refresh_probe()
        self.dialog.set_backend(self.backend, self.gpu_index)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def refresh_probe(self):
        self.probe = probe_environment()
        if hasattr(self, "dialog"):
            self.dialog.show_probe(self.probe, self.backend)
        self._sync_compact_label()
        return self.probe

    def apply_selection(self):
        if self.deploy._job_controller.state in (
            JobState.STARTING,
            JobState.RUNNING,
            JobState.STOPPING,
        ):
            self.dialog.selection_status.setText(
                "Stop the current deployment before changing execution backend."
            )
            return False

        backend = self.dialog.backend_combo.currentData()
        self.backend = normalize_backend(backend, self.deploy.useCPU)
        self.gpu_index = int(self.dialog.gpu_index.value())
        recommendation = self.probe.get("recommended", "cpu")
        if self.backend == "cpu":
            self.deploy.useCPU = True
        elif self.backend in ("cuda", "tensorrt"):
            self.deploy.useCPU = False
        else:
            self.deploy.useCPU = recommendation == "cpu"

        self._original_update_session_config()
        self._persist_backend_fields()
        self._sync_compact_label()
        self.dialog.show_probe(self.probe, self.backend)
        return True

    def _persist_backend_fields(self):
        config = _read_json(self.config_path)
        if not config:
            return
        config["execution_backend"] = self.backend
        config["execution_backend_gpu_index"] = int(self.gpu_index)
        config["useCPU"] = "CPU" if self.deploy.useCPU else "GPU"
        _write_json_atomic(self.config_path, config)

    def _sync_compact_label(self):
        backend = self.backend.upper()
        state = ""
        if self.probe:
            state, _ = backend_status(self.probe, self.backend)
        suffix = f" • {state}" if state else ""
        self.deploy.docker_button.setText(f"Backend  •  {backend}{suffix}")
        self.header_button.setText(f"Performance • {backend}")

    def run_benchmark(self):
        if self._benchmark_thread is not None and self._benchmark_thread.is_alive():
            return
        if self.deploy._job_controller.state in (
            JobState.STARTING,
            JobState.RUNNING,
            JobState.STOPPING,
        ):
            self.dialog.benchmark_output.setPlainText(
                "Stop normal Deploy before running deterministic backend benchmark."
            )
            return

        backends = self.dialog.benchmark_combo.currentData()
        fixture = self.package_root / "fixtures" / "p8_tracking.json"
        script = self.package_root / "scripts" / "epd_backend_benchmark.py"
        if not fixture.is_file() or not script.is_file():
            self.dialog.benchmark_output.setPlainText(
                "Benchmark fixture/script is unavailable in this checkout."
            )
            return

        command = [
            sys.executable,
            str(script),
            "--backends",
            str(backends),
            "--fixture",
            str(fixture),
            "--gpu-index",
            str(self.gpu_index),
        ]
        self.dialog.benchmark_button.setEnabled(False)
        self.dialog.benchmark_output.setPlainText(
            "Running deterministic replay benchmark…\n"
            "This can take several minutes because each backend must pass replay acceptance."
        )

        def worker():
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=900,
                )
                payload = _parse_last_json(result.stdout)
                if payload is None:
                    self.events.put((
                        "benchmark_error",
                        (result.stderr or result.stdout)[-5000:],
                    ))
                else:
                    self.events.put(("benchmark", payload))
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.events.put(("benchmark_error", str(exc)))

        self._benchmark_thread = threading.Thread(
            target=worker,
            daemon=True,
            name="EPDBackendBenchmark",
        )
        self._benchmark_thread.start()

    def _poll_events(self):
        processed = 0
        while processed < 10:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            processed += 1
            if kind == "benchmark":
                self._show_benchmark(payload)
                self.dialog.benchmark_button.setEnabled(True)
            elif kind == "benchmark_error":
                self.dialog.benchmark_output.setPlainText(
                    "Benchmark failed:\n" + str(payload))
                self.dialog.benchmark_button.setEnabled(True)

    def _show_benchmark(self, report):
        lines = []
        for record in report.get("records", []):
            backend = str(record.get("backend", "?")).upper()
            status = record.get("status", "UNKNOWN")
            perf = record.get("performance") or {}
            latency = perf.get("inference_latency_avg_ms")
            rate = perf.get("inference_rate_hz")
            wall = record.get("wall_seconds")
            equivalent = record.get("equivalent_to_cpu")
            bits = [f"{backend}: {status}"]
            if latency is not None:
                bits.append(f"avg inference {latency:.2f} ms")
            if rate is not None:
                bits.append(f"{rate:.2f} Hz")
            if wall is not None:
                bits.append(f"wall {wall:.2f} s")
            if equivalent is True:
                bits.append("CPU semantic match")
            elif equivalent is False:
                bits.append("CPU semantic mismatch/review")
            lines.append(" • ".join(bits))
            if record.get("error"):
                lines.append("  " + str(record["error"]))
            for warning in record.get("warnings", []):
                lines.append("  WARNING: " + str(warning))
        if not lines:
            lines = [json.dumps(report, indent=2, sort_keys=True)]
        self.dialog.benchmark_output.setPlainText("\n".join(lines))

    def shutdown(self):
        self.poll_timer.stop()
        self.label_timer.stop()


def apply_performance_backends(main_window):
    existing = getattr(main_window, "_epd8_performance_backends", None)
    if existing is not None:
        return existing
    controller = PerformanceBackendsController(main_window)
    main_window._epd8_performance_backends = controller
    return controller
