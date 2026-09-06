"""EPD-6 productization entry point for 3D perception diagnostics."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication

from windows.three_d_diagnostics import ThreeDInspectorController


def apply_epd6_productization(main_window):
    """Install the read-only 3D Inspector once on the current launcher."""
    if getattr(main_window, "_epd6_productization_applied", False):
        return getattr(main_window, "_epd6_3d_inspector", None)

    main_window._epd6_productization_applied = True
    controller = ThreeDInspectorController(main_window)
    main_window._epd6_3d_inspector = controller

    shortcut = QShortcut(QKeySequence("Ctrl+Shift+3"), main_window.deploy_window)
    shortcut.setContext(Qt.WindowShortcut)
    shortcut.activated.connect(controller.show)
    controller.shortcut = shortcut

    app = QApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(controller.shutdown)
    return controller
