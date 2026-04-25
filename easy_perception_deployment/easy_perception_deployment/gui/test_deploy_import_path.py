import importlib
import sys
import types
from pathlib import Path


def _install_stub_modules(monkeypatch):
    qtcore = types.ModuleType("PySide6.QtCore")
    qtcore.QObject = type("QObject", (), {})
    qtcore.QSize = type("QSize", (), {})
    qtcore.QThread = type("QThread", (), {})
    qtcore.QTimer = type("QTimer", (), {})
    qtcore.QElapsedTimer = type("QElapsedTimer", (), {})
    qtcore.Signal = lambda *args, **kwargs: object()
    qtcore.Slot = lambda *args, **kwargs: (lambda fn: fn)

    qtgui = types.ModuleType("PySide6.QtGui")
    qtgui.QIcon = type("QIcon", (), {})

    qtwidgets = types.ModuleType("PySide6.QtWidgets")
    for class_name in (
        "QComboBox",
        "QFileDialog",
        "QGridLayout",
        "QLabel",
        "QMessageBox",
        "QPushButton",
        "QWidget",
        "QDoubleSpinBox",
        "QSpinBox",
    ):
        setattr(qtwidgets, class_name, type(class_name, (), {}))

    pyside6 = types.ModuleType("PySide6")

    monkeypatch.setitem(sys.modules, "PySide6", pyside6)
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", qtcore)
    monkeypatch.setitem(sys.modules, "PySide6.QtGui", qtgui)
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", qtwidgets)

    counting = types.ModuleType("windows.Counting")
    counting.CountingWindow = type("CountingWindow", (), {})
    tracking = types.ModuleType("windows.Tracking")
    tracking.TrackingWindow = type("TrackingWindow", (), {})
    job_controller = types.ModuleType("windows.job_controller")
    job_controller.JobController = type("JobController", (), {})
    job_controller.JobState = type("JobState", (), {})

    monkeypatch.setitem(sys.modules, "windows.Counting", counting)
    monkeypatch.setitem(sys.modules, "windows.Tracking", tracking)
    monkeypatch.setitem(sys.modules, "windows.job_controller", job_controller)


def test_deploy_import_uses_package_level_scripts(monkeypatch):
    gui_dir = Path(__file__).resolve().parent
    package_root = gui_dir.parent

    monkeypatch.chdir(gui_dir)
    _install_stub_modules(monkeypatch)

    monkeypatch.setattr(
        sys,
        "path",
        [str(gui_dir)] + [p for p in sys.path if p not in {str(gui_dir), str(package_root)}],
    )

    for name in ("windows.Deploy", "scripts", "scripts.cli", "scripts.cli.config_schema"):
        sys.modules.pop(name, None)

    deploy_module = importlib.import_module("windows.Deploy")
    schema_module = importlib.import_module("scripts.cli.config_schema")

    assert deploy_module._SCHEMA_IMPORT_ROOT == str(package_root)
    assert sys.path[0] == str(package_root)
    assert Path(schema_module.__file__).resolve().is_relative_to(
        (package_root / "scripts").resolve()
    )
