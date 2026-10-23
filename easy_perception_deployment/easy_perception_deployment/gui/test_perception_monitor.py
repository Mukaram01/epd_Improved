# Copyright 2026 Easy Perception Deployment contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Perception Monitor: readiness parsing, replay and the Deploy workspace.

Readiness messages below use the exact key/value format written by
EPD::makeReadinessReport. Session tests use the bundled SYNTHETIC fixtures.
"""

import gc
import os
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QPushButton

from windows.perception_readiness import (
    STATES,
    ReplayPlayer,
    ReplayState,
    SessionError,
    associate,
    format_stamp,
    live_health,
    load_session,
    parse_readiness_message,
    replay_health,
    stale_after_s,
)

FIXTURES = (Path(__file__).resolve().parents[1] / "fixtures" /
            "perception_sessions")
ALL_STATES = FIXTURES / "synthetic_readiness_states"
NO_READINESS = FIXTURES / "synthetic_no_readiness"
STAMP = 1792178912205559570


def _status(index, **values):
    defaults = {
        "state": "GRASP_CANDIDATE", "class": "banana",
        "confidence": "0.777938", "roi": "570,332,122,161",
        "mask_pixels": "8568", "valid_depth_pixels": "8392",
        "position_valid": "true", "orientation_valid": "true",
        "centroid_m": "-0.00141905,0.0208204,0.499185",
        "axis": "0.610761,0.789841,-0.055886", "reasons": "none",
        "note": "", "track_id": "7", "observation_id": "6",
    }
    defaults.update(values)
    return SimpleNamespace(
        name=f"easy_perception_deployment/detection/{index}",
        level=b"\x00", message=defaults["state"], hardware_id="camera",
        values=[SimpleNamespace(key=k, value=v) for k, v in
                defaults.items()])


def _message(statuses, stamp_ns=STAMP, frame="camera_color_optical_frame"):
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=stamp_ns // 10**9,
                                  nanosec=stamp_ns % 10**9),
            frame_id=frame),
        status=statuses)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


# --------------------------------------------------------------- contract
def test_parse_keeps_reported_identity_geometry_and_reasons():
    report = parse_readiness_message(_message([
        _status(0, state="LOCALISED", **{"class": "apple"},
                confidence="0.938519", orientation_valid="false",
                centroid_m="0.0420817,0.00106492,0.473662", axis="",
                reasons="degenerate_axis,ambiguous_axis_no_pose",
                track_id="6"),
        _status(1),
    ]))
    assert report.stamp_ns == STAMP
    assert format_stamp(report.stamp_ns) == "1792178912.205559570"
    assert report.frame_id == "camera_color_optical_frame"
    assert report.observation_id == "6"
    first, second = report.records
    assert (first.index, first.class_name, first.state) == (
        0, "apple", "LOCALISED")
    assert first.confidence == pytest.approx(0.938519)
    assert first.track_id == "6"
    assert first.centroid_m == pytest.approx((0.0420817, 0.00106492,
                                              0.473662))
    assert first.axis is None
    assert first.reasons == ("degenerate_axis", "ambiguous_axis_no_pose")
    assert first.level == 0
    assert first.issues == []
    assert second.axis == pytest.approx((0.610761, 0.789841, -0.055886))
    assert second.reasons == ()


def test_parse_never_shows_geometry_beyond_the_reported_state():
    report = parse_readiness_message(_message([
        _status(0, state="DETECTED", position_valid="false",
                orientation_valid="false", reasons="insufficient_depth"),
        _status(1, state="BOGUS"),
        _status(2, track_id="", confidence=""),
    ]))
    detected, bogus, untracked = report.records
    assert detected.centroid_m is None and detected.axis is None
    assert any("not shown" in issue for issue in detected.issues)
    assert not bogus.state_known
    assert any("Unrecognised" in issue for issue in bogus.issues)
    assert untracked.track_id is None
    assert untracked.confidence is None
    assert untracked.key == ("proposal", 2)


def test_association_requires_identical_source_header():
    report = parse_readiness_message(_message([_status(0)]))
    frame = "camera_color_optical_frame"
    assert associate(STAMP, frame, report).matched
    newer = associate(STAMP + 1, frame, report)
    assert not newer.matched and "do not belong together" in newer.reason
    assert not associate(STAMP, "other_frame", report).matched
    assert not associate(None, frame, report).matched
    assert not associate(STAMP, frame, None).matched


# ---------------------------------------------------------------- sessions
def test_synthetic_fixture_covers_every_state_and_failure_case():
    session = load_session(ALL_STATES)
    assert session.is_synthetic
    states = {record.state for record in session.frames[0].readiness.records}
    assert states == set(STATES)
    labels = [frame.label for frame in session.frames]
    assert labels[2] == "Readiness report missing"
    assert session.frames[2].readiness is None
    assert not session.frames[3].association().matched
    assert session.frames[4].image_path is None
    assert session.frames[5].readiness.records == []
    assert session.report_rate_hz() == pytest.approx(2.0)


def test_tampered_or_unlabelled_recordings_are_refused(tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(ALL_STATES, copy)
    with open(copy / "frames" / "000.png", "ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(SessionError, match="SHA256"):
        load_session(copy)

    manifest = (ALL_STATES / "session.json").read_text(encoding="utf-8")
    unlabelled = tmp_path / "unlabelled"
    shutil.copytree(ALL_STATES, unlabelled)
    (unlabelled / "session.json").write_text(
        manifest.replace('"kind": "synthetic"', '"kind": "camera"'),
        encoding="utf-8")
    with pytest.raises(SessionError, match="recorded or synthetic"):
        load_session(unlabelled)

    escaped = tmp_path / "escaped"
    shutil.copytree(ALL_STATES, escaped)
    (escaped / "session.json").write_text(
        manifest.replace('"frames/000.png"', '"../000.png"'),
        encoding="utf-8")
    with pytest.raises(SessionError, match="inside the recording"):
        load_session(escaped)

    with pytest.raises(SessionError):
        load_session(tmp_path / "missing")


# ------------------------------------------------------------------ replay
def test_player_follows_source_spacing_and_compresses_long_gaps():
    clock = FakeClock()
    player = ReplayPlayer(max_gap_s=1.5, clock=clock)
    player.load([0, 600_000_000, 1_200_000_000, 300_000_000_000])
    assert player.compressed_gaps == 1
    player.play()
    assert player.index == 0
    clock.now += 0.61
    assert player.tick() == 1
    player.pause()
    clock.now += 10.0
    assert player.tick() == 1 and player.state == ReplayState.PAUSED
    player.play()
    clock.now += 0.6
    assert player.tick() == 2
    clock.now += 1.5
    assert player.tick() == 3
    clock.now += 5.0
    player.tick()
    assert player.state == ReplayState.ENDED and player.index == 3
    player.restart()
    assert player.state == ReplayState.PLAYING and player.index == 0
    player.stop()
    assert player.state == ReplayState.STOPPED and player.index is None
    assert player.step(1) is None
    player.play()
    player.pause()
    assert player.step(1) == 1 and player.step(-5) == 0


# ------------------------------------------------------------------ health
def test_live_health_states_are_truthful():
    def health(job, error="", publishers=1, age=None, running=10.0):
        return live_health(job, error, publishers, age, 5.0, running).state

    assert health("STOPPED") == "STOPPED"
    assert health("STOPPED", age=1.0) == "LIVE"
    assert health("STOPPED", age=9.0) == "STALE"
    assert health("STARTING") == "STARTING"
    assert health("RUNNING", publishers=1) == "WAITING"
    assert health("RUNNING", publishers=0, running=2.0) == "WAITING"
    assert health("RUNNING", publishers=0, running=6.0) == "UNAVAILABLE"
    assert health("RUNNING", age=1.0) == "LIVE"
    assert health("RUNNING", age=12.0) == "STALE"
    assert health("FAILED") == "FAILED"
    assert health("RUNNING", error="no rclpy") == "UNAVAILABLE"
    assert stale_after_s([0.0, 4.0, 8.0]) == pytest.approx(12.0)
    assert stale_after_s([]) == 5.0


def test_replay_health_marks_end_of_replay_stale():
    session = load_session(ALL_STATES)
    frame = session.frames[0]
    assert replay_health(ReplayState.PLAYING, session, frame).state == (
        "REPLAY ACTIVE")
    assert replay_health(ReplayState.ENDED, session, frame).state == "STALE"
    assert replay_health(ReplayState.STOPPED, session, None).state == (
        "STOPPED")
    failed = replay_health(ReplayState.EMPTY, None, None, "bad file")
    assert failed.state == "FAILED"


# --------------------------------------------------------- full Deploy UI
def _build_epd_window():
    """Apply every layer in main.py's order (checked by the test below)."""
    from windows.Main import MainWindow
    from windows.acceptance_stability import (
        apply_deploy_ui_stability, install_ros_executor_stability)
    from windows.deploy_runtime_truth import apply_deploy_runtime_truth
    from windows.epd0_productization import apply_epd0_productization
    from windows.epd1_productization import apply_epd1_productization
    from windows.epd2_productization import apply_epd2_productization
    from windows.epd3_productization import apply_epd3_productization
    from windows.epd4_productization import apply_epd4_productization
    from windows.epd5_integration import finalize_epd5_integration
    from windows.epd5_productization import apply_epd5_productization
    from windows.epd6_productization import apply_epd6_productization
    from windows.epd8_productization import apply_epd8_productization
    from windows.epd9_productization import apply_epd9_productization
    from windows.model_output_contract import apply_model_output_contract
    from windows.perception_monitor import apply_perception_monitor

    install_ros_executor_stability()
    apply_model_output_contract()
    window = MainWindow()
    apply_deploy_ui_stability(window)
    apply_epd0_productization(window)
    apply_deploy_runtime_truth(window)
    apply_epd1_productization(window)
    apply_epd2_productization(window)
    apply_epd3_productization(window)
    apply_epd4_productization(window)
    epd5 = apply_epd5_productization(window)
    finalize_epd5_integration(window, epd5)
    apply_epd6_productization(window)
    apply_epd8_productization(window)
    apply_epd9_productization(window)
    apply_perception_monitor(window)
    return window


@pytest.fixture(scope="module")
def epd_app(qapp, tmp_path_factory):
    """One full application per module, in its own copy of the package.

    Module fixtures start before conftest's per-test sandbox, so this one
    sandboxes itself the same way: Deploy writes its config on start.
    """
    import conftest

    root = tmp_path_factory.mktemp("monitor") / "easy_perception_deployment"
    shutil.copytree(conftest._PRISTINE_ROOT, root, symlinks=True)
    gui_dir = root / "gui"
    with pytest.MonkeyPatch.context() as patch:
        for cls in conftest._SANDBOXED_WINDOW_CLASSES:
            patch.setattr(cls, "_MODULE_DIR", gui_dir / "windows")
            patch.setattr(cls, "_GUI_DIR", gui_dir)
            if hasattr(cls, "_PACKAGE_ROOT"):
                patch.setattr(cls, "_PACKAGE_ROOT", root)
        patch.chdir(gui_dir)
        window = _build_epd_window()
        window.deployPackage()
        deploy = window.deploy_window
        for _ in range(20):
            qapp.processEvents()
        yield SimpleNamespace(window=window, deploy=deploy,
                              monitor=deploy.perception_monitor, root=root)
        deploy.perception_monitor.shutdown()
        for top in QApplication.topLevelWidgets():
            top.close()
        qapp.processEvents()
        # Collect the window's wrappers here, on the GUI thread.
        del window, deploy
        gc.collect()
        qapp.processEvents()


@pytest.fixture
def epd(epd_app, qtbot):
    deploy, monitor = epd_app.deploy, epd_app.monitor
    if not deploy.isVisible():
        epd_app.window.deployPackage()
    qtbot.waitExposed(deploy)
    deploy.resize(QSize(1100, 760))
    monitor.live_button.click()
    monitor.load_error = ""
    monitor.session = None
    monitor.player.unload()
    clock = FakeClock()
    monitor._clock = clock
    monitor.player._clock = clock
    yield SimpleNamespace(window=epd_app.window, deploy=deploy,
                          monitor=monitor, clock=clock)
    monitor.stop()
    for top in QApplication.topLevelWidgets():
        if top.isVisible() and top not in (deploy, epd_app.window):
            top.close()


def _rows(monitor):
    model = monitor.table_model
    return [[model.text(row, column) for column in range(model.columnCount())]
            for row in range(model.rowCount())]


def test_monitor_app_runs_in_a_package_copy(epd_app):
    package = Path(__file__).resolve().parents[1]
    assert epd_app.deploy._path_to_session_config.startswith(
        str(epd_app.root))
    assert not epd_app.deploy._path_to_session_config.startswith(
        str(package) + os.sep)


def test_main_installs_perception_monitor_last():
    main_py = Path(__file__).with_name("main.py").read_text(encoding="utf-8")
    assert (main_py.index("apply_epd9_productization(window1)") <
            main_py.index("apply_perception_monitor(window1)") <
            main_py.index("window1.show()"))


def test_workspace_is_preview_first_and_keeps_existing_controls(epd):
    monitor, deploy = epd.monitor, epd.deploy
    tabs = [monitor.side_tabs.tabText(i)
            for i in range(monitor.side_tabs.count())]
    assert tabs == ["Inspector", "Health", "Setup"]
    for widget in (deploy.model_button, deploy.list_button,
                   deploy.topic_button, deploy.usecase_config_button,
                   deploy.confidence_spinbox, deploy.max_detections_spinbox):
        assert monitor.setup_scroll.isAncestorOf(widget)
    assert deploy.run_button.isVisible()
    assert monitor.view_stack.currentWidget() is monitor.preview_card
    labels = {button.text() for button in deploy.findChildren(QPushButton)}
    assert {"Profiles && Replay", "3D Inspector"} <= labels


def test_live_mode_shows_readiness_unavailable_not_an_empty_table(epd):
    monitor = epd.monitor
    assert monitor.mode == monitor.MODE_LIVE
    assert monitor.table_stack.currentWidget() is monitor.unavailable_label
    assert "READINESS UNAVAILABLE" in monitor.unavailable_label.text()
    assert monitor.table_model.rowCount() == 0
    assert monitor.health_badge.text() in ("STOPPED", "UNAVAILABLE")


def test_replay_shows_all_five_states_with_recorded_values(epd):
    monitor = epd.monitor
    assert monitor.open_session(str(ALL_STATES))
    assert monitor.view_stack.currentWidget() is not monitor.preview_card
    assert "SYNTHETIC" in monitor.provenance_label.text()
    monitor.play()
    assert monitor.health_badge.text() == "REPLAY ACTIVE"
    rows = _rows(monitor)
    assert [row[4] for row in rows] == [
        "GRASP_CANDIDATE", "LOCALISED", "POSE_VALID", "DETECTED",
        "REJECTED"]
    assert [row[3] for row in rows] == ["1", "2", "3", "4", "5"]
    assert rows[0][6] == "0.081, -0.032, 0.512"
    assert rows[3][6] == "—" and rows[4][6] == "—"
    assert rows[4][5] == "Overlapping ownership"
    assert rows[2][5] == "Not a target class"
    assert rows[0][7] == "…1000.000 s"
    assert monitor.view.draws_rois()


def test_inspector_shows_evidence_while_paused(epd):
    monitor = epd.monitor
    monitor.open_session(str(ALL_STATES))
    monitor.play()
    monitor.pause()
    monitor.select_row(4)
    values = monitor.inspector.values
    assert monitor.side_tabs.currentWidget() is monitor.inspector
    assert monitor.inspector.title.text() == "#4 · banana"
    assert "REJECTED" in values["readiness"].text()
    assert "overlapping_ownership" in values["reasons"].text()
    assert values["dimensions"].text().startswith("Unavailable")
    assert values["position"].text().startswith("Not valid")
    assert values["association"].text().startswith("Matched")
    assert "not usable" in monitor.inspector.usable.text()
    assert monitor.view._selected_key == ("track", "5")
    epd.clock.now += 30.0
    monitor._replay_tick()
    assert monitor.player.state == ReplayState.PAUSED
    assert monitor.inspector.title.text() == "#4 · banana"
    monitor.select_row(0)
    assert "GRASP_CANDIDATE" in values["readiness"].text()
    assert "not checked here" in values["downstream"].text()
    assert "orientation valid" in monitor.inspector.usable.text()


def test_selection_follows_track_ids_across_frames(epd):
    monitor = epd.monitor
    monitor.open_session(str(ALL_STATES))
    monitor.play()
    monitor.select_row(1)
    epd.clock.now += 0.55
    monitor._replay_tick()
    assert monitor.player.index == 1
    assert monitor.selected_key == ("track", "2")
    assert monitor.inspector.title.text() == "#1 · banana"


def test_mismatched_image_and_report_are_never_overlaid(epd):
    monitor = epd.monitor
    monitor.open_session(str(ALL_STATES))
    for _ in range(4):
        monitor.step(1)
    assert monitor.player.index == 3
    assert monitor.table_model.rowCount() == 5
    assert not monitor.view.draws_rois()
    assert "do not belong together" in monitor.view._notice
    monitor.select_row(0)
    assert monitor.inspector.values["association"].text().startswith(
        "Not matched")
    assert any("do not belong together" in message
               for message in monitor.health_state.messages)


def test_missing_readiness_rgb_and_empty_reports(epd):
    monitor = epd.monitor
    monitor.open_session(str(ALL_STATES))
    for _ in range(3):
        monitor.step(1)
    assert monitor.table_stack.currentWidget() is monitor.unavailable_label
    assert "No readiness report was recorded for this frame" in (
        monitor.unavailable_label.text())
    monitor.step(2)
    assert monitor.view._image is None
    assert "RGB frame missing" in monitor.view._placeholder
    assert monitor.table_model.rowCount() == 5
    monitor.step(1)
    assert monitor.table_stack.currentWidget() is monitor.table
    assert monitor.table_model.rowCount() == 0
    assert monitor.summary_label.text() == (
        "Report received with 0 detections")

    assert monitor.open_session(str(NO_READINESS))
    monitor.play()
    assert "This recording has no readiness topic" in (
        monitor.unavailable_label.text())
    assert monitor.health.values["readiness"].text() == (
        "Unavailable in this recording")


def test_transport_play_pause_restart_stop_and_end(epd):
    monitor, clock = epd.monitor, epd.clock
    monitor.open_session(str(ALL_STATES))
    assert monitor.health_badge.text() == "STOPPED"
    monitor.play()
    assert monitor.replay_timer.isActive()
    clock.now += 1.05
    monitor._replay_tick()
    assert monitor.player.index == 2
    monitor.pause()
    assert not monitor.replay_timer.isActive()
    assert not monitor.pause_button.isEnabled()
    monitor.restart()
    assert monitor.player.index == 0
    clock.now += 10.0
    monitor._replay_tick()
    assert monitor.player.state == ReplayState.ENDED
    assert monitor.health_badge.text() == "STALE"
    assert monitor.view._stale
    assert not monitor.replay_timer.isActive()
    monitor.stop()
    assert monitor.health_badge.text() == "STOPPED"
    assert monitor.table_model.rowCount() == 0
    assert "Replay stopped" in monitor.unavailable_label.text()
    assert monitor.inspector.record is None


def test_replay_timer_drives_playback_in_real_time(epd, qtbot):
    monitor = epd.monitor
    monitor._clock = time.monotonic
    monitor.player._clock = time.monotonic
    monitor.open_session(str(ALL_STATES))
    monitor.play()
    qtbot.waitUntil(lambda: monitor.player.index == 1, timeout=3000)
    qtbot.waitUntil(lambda: monitor.player.state == ReplayState.ENDED,
                    timeout=6000)
    assert not monitor.replay_timer.isActive()


def test_tampered_recording_fails_visibly(epd, tmp_path):
    copy = tmp_path / "tampered"
    shutil.copytree(ALL_STATES, copy)
    with open(copy / "frames" / "001.png", "ab") as stream:
        stream.write(b"x")
    monitor = epd.monitor
    assert monitor.open_session(str(copy))
    monitor.play()
    assert monitor.view._image is not None
    epd.clock.now += 0.55
    monitor._replay_tick()
    assert monitor.session is None
    assert monitor.health_badge.text() == "FAILED"
    assert "SHA256" in monitor.session_label.text()
    assert monitor.view._image is None
    assert not monitor.play_button.isEnabled()
    assert not monitor.replay_timer.isActive()


def test_switching_session_or_mode_clears_stale_details(epd):
    monitor = epd.monitor
    monitor.open_session(str(ALL_STATES))
    monitor.play()
    monitor.select_row(0)
    assert monitor.selected_key is not None
    monitor.open_session(str(NO_READINESS))
    assert monitor.selected_key is None
    assert monitor.inspector.record is None
    monitor.live_button.click()
    assert monitor.mode == monitor.MODE_LIVE
    assert monitor.player.state == ReplayState.STOPPED
    assert monitor.table_model.rowCount() == 0


def test_repeated_replay_cycles_release_resources(epd):
    monitor = epd.monitor
    baseline = threading.active_count()
    for _ in range(20):
        monitor.replay_button.click()
        assert monitor.open_session(str(ALL_STATES))
        monitor.play()
        for _ in range(6):
            epd.clock.now += 0.5
            monitor._replay_tick()
        monitor.select_row(0)
        monitor.stop()
        monitor.live_button.click()
    assert len(monitor.cache) == 0
    assert not monitor.replay_timer.isActive()
    monitor.replay_button.click()
    assert not monitor.live_subscribed()
    assert len(monitor.cache._images) <= monitor.cache._capacity
    assert threading.active_count() <= baseline + 1
    epd.deploy.close()
    assert not monitor.workers_alive()
    assert not monitor.live_timer.isActive()


def test_layout_fits_1366x768(epd, qtbot):
    deploy, monitor = epd.deploy, epd.monitor
    deploy.resize(QSize(1366, 768))
    monitor.open_session(str(ALL_STATES))
    monitor.play()
    qtbot.wait(50)
    width = deploy.width()
    for button in (monitor.live_button, monitor.replay_button,
                   monitor.play_button, monitor.pause_button,
                   monitor.restart_button, monitor.stop_button,
                   deploy.run_button):
        assert button.isVisible()
        assert button.width() >= button.minimumSizeHint().width()
        right = button.mapTo(deploy, button.rect().topRight()).x()
        assert right <= width
    assert monitor.open_button.width() >= (
        monitor.open_button.minimumSizeHint().width())
    assert monitor.view.width() >= 480 and monitor.view.height() >= 180
    assert monitor.table.viewport().height() >= 3 * 24
    assert monitor.side_tabs.width() >= 340


def test_run_button_keeps_compact_icon_across_job_states(epd):
    from windows.job_controller import JobState

    deploy = epd.deploy
    controller = deploy._job_controller
    controller.set_state(JobState.STARTING)
    controller.set_state(JobState.IDLE)
    assert deploy.run_button.iconSize() == QSize(24, 24)
    assert (deploy.run_button.width() >=
            deploy.run_button.minimumSizeHint().width())


def test_profiles_and_replay_dialog_still_opens(epd, qtbot):
    button = next(b for b in epd.deploy.findChildren(QPushButton)
                  if b.text() == "Profiles && Replay")
    button.click()
    dialogs = [top for top in QApplication.topLevelWidgets()
               if top.windowTitle() == "EPD Profiles & Replay" and
               top.isVisible()]
    assert len(dialogs) == 1
    dialogs[0].close()


# --------------------------------------------------- ROS paths (local only)
def test_exporter_pairs_reports_with_exact_rgb_frames(tmp_path):
    pytest.importorskip("rosbag2_py")
    pytest.importorskip("cv2")
    import importlib.util
    import rosbag2_py
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
    from diagnostic_msgs.msg import KeyValue
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Image

    script = (Path(__file__).resolve().parents[1] / "scripts" /
              "epd_perception_session_export.py")
    spec = importlib.util.spec_from_file_location("exporter", script)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    bag = tmp_path / "bag"
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    topic = "/camera/color/image_raw"
    writer.create_topic(rosbag2_py.TopicMetadata(
        name=topic, type="sensor_msgs/msg/Image",
        serialization_format="cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(
        name=exporter.READINESS_TOPIC,
        type="diagnostic_msgs/msg/DiagnosticArray",
        serialization_format="cdr"))
    originals = []
    for index, stamp in enumerate((5_000_000_000, 5_500_000_000)):
        image = Image(height=2, width=3, encoding="rgb8", step=9,
                      data=bytes(range(index, index + 18)))
        image.header.stamp.sec, image.header.stamp.nanosec = divmod(
            stamp, 10**9)
        image.header.frame_id = "cam"
        report = DiagnosticArray()
        report.header.stamp.sec = image.header.stamp.sec
        # The second report has no exactly matching RGB frame.
        report.header.stamp.nanosec = image.header.stamp.nanosec + index
        report.header.frame_id = "cam"
        report.status = [DiagnosticStatus(
            level=b"\x02", name="easy_perception_deployment/detection/0",
            message="REJECTED", hardware_id="cam",
            values=[KeyValue(key="state", value="REJECTED"),
                    KeyValue(key="reasons", value="overlapping_ownership")])]
        originals.append(report)
        writer.write(topic, serialize_message(image), stamp)
        writer.write(exporter.READINESS_TOPIC, serialize_message(report),
                     stamp + 1)
    del writer

    output = tmp_path / "session"
    exporter.export(bag, output, topic, "unit", "test bag", 100)
    session = load_session(output)
    assert session.kind == "recorded"
    assert session.frames[0].association().matched
    assert session.frames[1].image_path is None
    from windows.perception_readiness import diagnostic_array_to_dict
    import json
    manifest = json.loads((output / "session.json").read_text())
    assert [frame["readiness"] for frame in manifest["frames"]] == [
        diagnostic_array_to_dict(report) for report in originals]
    with pytest.raises(SystemExit):
        exporter.export(bag, output, topic, "", "", 100)


def test_live_monitor_consumes_the_production_topic(monkeypatch):
    rclpy = pytest.importorskip("rclpy")
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
    from diagnostic_msgs.msg import KeyValue
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from windows.perception_monitor import ReadinessTopicMonitor
    from windows.perception_readiness import READINESS_TOPIC

    monkeypatch.setenv("ROS_DOMAIN_ID", str(150 + os.getpid() % 50))
    monitor = ReadinessTopicMonitor(max_pending=2)
    monitor.start()
    context = rclpy.context.Context()
    context.init(args=None)
    try:
        node = rclpy.create_node("readiness_test_pub", context=context)
        publisher = node.create_publisher(
            DiagnosticArray, READINESS_TOPIC,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        message = DiagnosticArray()
        message.header.stamp.sec, message.header.stamp.nanosec = 12, 34
        message.header.frame_id = "cam"
        message.status = [DiagnosticStatus(
            level=b"\x01", name="easy_perception_deployment/detection/3",
            message="LOCALISED", hardware_id="cam",
            values=[KeyValue(key="state", value="LOCALISED"),
                    KeyValue(key="class", value="apple"),
                    KeyValue(key="position_valid", value="true"),
                    KeyValue(key="centroid_m", value="0.1,0.2,0.3"),
                    KeyValue(key="track_id", value="")])]
        reports = []
        deadline = time.monotonic() + 15.0
        while not reports and time.monotonic() < deadline:
            publisher.publish(message)
            time.sleep(0.2)
            reports = monitor.drain()["reports"]
        assert reports, "readiness report not received"
        report = reports[-1]
        assert report.stamp_ns == 12_000_000_034
        record = report.records[0]
        assert (record.index, record.state, record.class_name) == (
            3, "LOCALISED", "apple")
        assert record.centroid_m == pytest.approx((0.1, 0.2, 0.3))
        assert monitor.drain()["publisher_count"] >= 1
        for _ in range(10):
            publisher.publish(message)
        time.sleep(1.0)
        assert len(monitor.drain()["reports"]) <= 2

        # Leaving live mode drops the subscription but keeps the context.
        thread = monitor._thread
        monitor.set_active(False)
        deadline = time.monotonic() + 5.0
        while monitor.subscribed() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not monitor.subscribed()
        publisher.publish(message)
        time.sleep(0.5)
        assert monitor.drain()["reports"] == []
        monitor.set_active(True)
        deadline = time.monotonic() + 5.0
        while not monitor.subscribed() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert monitor.subscribed() and monitor._thread is thread
        node.destroy_node()
    finally:
        context.shutdown()
        assert monitor.stop()
    assert not monitor.is_running()
