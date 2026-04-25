import windows.Deploy as deploy_module
from windows.Deploy import DeployWindow


def test_fps_monitor_disabled_by_env(qtbot, monkeypatch):
    monkeypatch.setenv('EPD_DISABLE_FPS_MONITOR', '1')

    widget = DeployWindow(True)
    qtbot.addWidget(widget)

    assert widget._fps_monitor is None
    assert widget._fps_poll_timer is None
    assert widget.fps_label.text() == 'FPS: disabled'


def test_fps_label_polled_with_qtimer(qtbot, monkeypatch):
    class _DummyMonitor:
        def __init__(self, mode):
            self.mode = mode

        def start(self):
            return None

        def stop(self):
            return None

        def wait(self, timeout_ms=None):
            return True

        def set_usecase_mode(self, mode):
            self.mode = mode

        def get_latest_text(self):
            return 'FPS: 17.5 | Latency: 10.0 ms'

    monkeypatch.delenv('EPD_DISABLE_FPS_MONITOR', raising=False)
    monkeypatch.setattr(deploy_module, '_RCLPY_AVAILABLE', True)
    monkeypatch.setattr(deploy_module, 'FPSMonitorThread', _DummyMonitor)

    widget = DeployWindow(True)
    qtbot.addWidget(widget)

    qtbot.waitUntil(
        lambda: widget.fps_label.text() == 'FPS: 17.5 | Latency: 10.0 ms',
        timeout=1000,
    )

    assert widget._fps_monitor is not None
    assert widget._fps_poll_timer is not None
    assert widget._fps_poll_timer.isActive()

    widget.shutdown()

    assert widget._fps_monitor is None
    assert widget._fps_poll_timer is None
