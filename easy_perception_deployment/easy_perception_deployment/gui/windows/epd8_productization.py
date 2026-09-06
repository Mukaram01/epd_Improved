"""EPD-8 productization entry point: performance backends and benchmarking."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication

from windows.backend_manager import apply_performance_backends


_HELP_TOPIC = """
<h2>Performance Backends</h2>
<p>EPD-8 makes execution-provider selection explicit instead of treating
<b>CPU/GPU</b> as a blind switch.</p>
<p>Open <b>Performance</b> from Deploy or press <b>Ctrl+Shift+B</b>.</p>
<ul>
<li><b>AUTO:</b> uses CUDA only when the deployment path can initialize it;
otherwise ONNX Runtime falls back to CPU.</li>
<li><b>CPU:</b> the reliable baseline and fallback.</li>
<li><b>CUDA:</b> explicit NVIDIA CUDA execution. A missing CUDA provider is a
blocking error rather than a silent CPU run.</li>
<li><b>TensorRT:</b> opt-in. It requires an ONNX Runtime vendor built with
TensorRT, EPD built with <code>-DEPD_ENABLE_TENSORRT=ON</code>, and an explicit
<code>EPD_TENSORRT_IMAGE</code> for Docker deployment.</li>
</ul>
<p>The manager probes architecture, Jetson markers, Docker, NVIDIA runtime,
container images and compiled provider capability where available.</p>
<h3>Benchmarking</h3>
<p>The benchmark reuses deterministic P8 replay. Speed is considered only after
replay PASS. Accelerated results are compared with the CPU semantic summary for
stable IDs, LOST lifecycle and geometry-quality totals.</p>
<h3>Jetson</h3>
<p>Jetson is treated as an NVIDIA aarch64 deployment target, not as a generic
x86 GPU. Use a JetPack-compatible native/vendor build or provide an explicitly
compatible <code>EPD_GPU_IMAGE</code>. EPD does not silently pull an x86 image and
claim Jetson support.</p>
"""


def _install_help_topic(main_window):
    help_window = getattr(main_window, "help_window", None)
    if help_window is None:
        return
    topic = "Performance Backends"
    if topic not in help_window.TOPICS:
        help_window.TOPICS[topic] = _HELP_TOPIC
    matches = help_window.topic_list.findItems(topic, Qt.MatchExactly)
    if not matches:
        help_window.topic_list.addItem(topic)


def apply_epd8_productization(main_window):
    """Install EPD-8 once without changing inference semantics or robot ownership."""
    if getattr(main_window, "_epd8_productization_applied", False):
        return getattr(main_window, "_epd8_performance_backends", None)
    main_window._epd8_productization_applied = True

    controller = apply_performance_backends(main_window)
    _install_help_topic(main_window)

    shortcut = QShortcut(QKeySequence("Ctrl+Shift+B"), main_window.deploy_window)
    shortcut.setContext(Qt.WindowShortcut)
    shortcut.activated.connect(controller.show)
    controller.shortcut = shortcut

    app = QApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(controller.shutdown)
    return controller
