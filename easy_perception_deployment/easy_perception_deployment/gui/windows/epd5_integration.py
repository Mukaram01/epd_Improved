"""Small integration hooks between EPD-5 and earlier productization layers."""

from __future__ import annotations

from types import MethodType

from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication


def finalize_epd5_integration(main_window, controller):
    """Wire EPD-5 to EPD-3 inspection and application shutdown."""
    if controller is None or getattr(controller, "_integration_finalized", False):
        return controller
    controller._integration_finalized = True

    original_sync = controller._sync_deploy

    def sync_and_reinspect(self, profile):
        original_sync(profile)
        model_manager = getattr(
            main_window,
            "_epd3_model_manager_controller",
            None,
        )
        if model_manager is not None:
            model_manager.schedule_inspection()
        deploy_ui = getattr(main_window, "_deploy_ui_controller", None)
        if deploy_ui is not None:
            deploy_ui.sync()

    controller._sync_deploy = MethodType(sync_and_reinspect, controller)

    shortcut = QShortcut(QKeySequence("Ctrl+Shift+P"), controller.deploy)
    shortcut.setContext(QtShortcutContext())
    shortcut.activated.connect(controller.show)
    controller._profiles_shortcut = shortcut

    app = QApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(controller.shutdown)
    return controller


def QtShortcutContext():
    """Return an application-safe shortcut context without importing Qt in callers."""
    from PySide6.QtCore import Qt

    return Qt.WindowShortcut
