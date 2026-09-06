"""Small integration hooks between EPD-5 and earlier productization layers."""

from __future__ import annotations

from types import MethodType

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication


_HELP_TOPIC = (
    "<h2>Profiles & Replay</h2>"
    "<p><b>Ctrl+Shift+P</b> opens EPD-5 from Deploy.</p>"
    "<p>A perception profile stores the model, labels, RGB topic, perception mode "
    "and runtime settings together instead of relying on three loose config files.</p>"
    "<p>Model and label SHA256 fingerprints are recorded when the files exist. "
    "Applying a profile refuses a different same-named asset.</p>"
    "<p><b>Known-good</b> is a workstation-local marker for the profile you want as "
    "your quick recovery configuration.</p>"
    "<h3>Deterministic replay</h3>"
    "<p>Stop normal Deploy first. Fixture replay starts its own production EPD node "
    "and reports the existing backend PASS/FAIL acceptance summary.</p>"
    "<h3>Rosbag replay</h3>"
    "<p>Apply the profile that matches the recording, start Deploy, inspect the bag, "
    "then play it. EPD-5 deliberately does not invent topic remappings.</p>"
)


def finalize_epd5_integration(main_window, controller):
    """Wire EPD-5 to model truth, Help, shortcuts and application shutdown."""
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
    shortcut.setContext(Qt.WindowShortcut)
    shortcut.activated.connect(controller.show)
    controller._profiles_shortcut = shortcut

    help_window = getattr(main_window, "help_window", None)
    if help_window is not None and "Profiles & Replay" not in help_window.TOPICS:
        help_window.TOPICS["Profiles & Replay"] = _HELP_TOPIC
        help_window.topic_list.addItem("Profiles & Replay")

    app = QApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(controller.shutdown)
    return controller
