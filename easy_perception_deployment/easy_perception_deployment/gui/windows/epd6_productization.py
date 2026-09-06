"""EPD-6 productization entry point for 3D perception diagnostics."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication

from windows.three_d_diagnostics import ThreeDInspectorController


_HELP_TOPIC = """
<h2>3D Perception Inspector</h2>
<p>EPD-6 adds a read-only inspector for <b>Localization</b> and <b>Tracking</b>.</p>
<p>Open it from Deploy with <b>3D Inspector</b> or <b>Ctrl+Shift+3</b>.</p>
<ul>
<li><b>3D health:</b> checks embedded depth shape, intrinsics, encoding and result/depth source timestamps.</li>
<li><b>Localized objects:</b> shows class, centroid, dimensions, point-cloud size, axis and a GUI-side sanity check.</li>
<li><b>Tracking IDs:</b> shows current stable IDs and LOST transitions from EPD tracking output.</li>
<li><b>Geometry diagnostics:</b> shows production valid/degraded/invalid geometry counters and failure reasons such as insufficient depth or invalid intrinsics.</li>
</ul>
<p>The sampled depth ratio is an estimate from the embedded depth image, not a full-frame statistic.</p>
<p>The inspector does not alter inference, filters, scene state, planning or robot motion. Plane/background filtering remains off until real workcell evidence justifies a specific filter.</p>
"""


def _install_help_topic(main_window):
    help_window = getattr(main_window, "help_window", None)
    if help_window is None:
        return
    topic = "3D Perception Inspector"
    if topic not in help_window.TOPICS:
        help_window.TOPICS[topic] = _HELP_TOPIC
    matches = help_window.topic_list.findItems(topic, Qt.MatchExactly)
    if not matches:
        help_window.topic_list.addItem(topic)


def apply_epd6_productization(main_window):
    """Install the read-only 3D Inspector once on the current launcher."""
    if getattr(main_window, "_epd6_productization_applied", False):
        return getattr(main_window, "_epd6_3d_inspector", None)

    main_window._epd6_productization_applied = True
    controller = ThreeDInspectorController(main_window)
    main_window._epd6_3d_inspector = controller
    _install_help_topic(main_window)

    shortcut = QShortcut(QKeySequence("Ctrl+Shift+3"), main_window.deploy_window)
    shortcut.setContext(Qt.WindowShortcut)
    shortcut.activated.connect(controller.show)
    controller.shortcut = shortcut

    app = QApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(controller.shutdown)
    return controller
