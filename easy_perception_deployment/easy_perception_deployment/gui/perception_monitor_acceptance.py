#!/usr/bin/env python3
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

"""Hardware-free acceptance of the Deploy Perception Monitor.

Starts the real application (gui/main.py with every layer), drives Deploy
through replay, inspection, failure cases and shutdown, saves screenshots
grabbed from the running windows and writes ``results.json``. No camera,
GPU or inference is used. Run from a sandbox copy of the package: Deploy
writes its config files on start.

  python3 perception_monitor_acceptance.py --session <recorded session> \\
      --out <dir> [--width 1366 --height 768 --cycles 20]

Optional ROS check: with ROS 2 sourced, the recorded readiness messages are
also published on the production topic (use an isolated ROS_DOMAIN_ID).
The process exit code is the verdict; results.json alone is not.
"""

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parent
FIXTURES = GUI_DIR.parent / "fixtures" / "perception_sessions"


class Acceptance:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.failed = False

    def check(self, scenario, condition, detail):
        status = "PASS" if condition else "FAIL"
        self.failed |= not condition
        self.results.append(
            {"scenario": scenario, "status": status, "detail": detail})
        print(f"[{status}] {scenario}: {detail}", flush=True)

    def shot(self, widget, name):
        path = self.out / f"{name}.png"
        widget.grab().save(str(path))
        print(f"[SHOT] {path}", flush=True)
        return path

    def write(self, extra):
        payload = {"results": self.results, "failed": self.failed}
        payload.update(extra)
        (self.out / "results.json").write_text(
            json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def _publish_recorded_reports(session_dir, seconds, stop):
    """Publish a recording's readiness messages on the production topic."""
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
    from diagnostic_msgs.msg import KeyValue
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    manifest = json.loads((Path(session_dir) / "session.json").read_text())
    messages = []
    for frame in manifest["frames"]:
        report = frame.get("readiness")
        if report is None:
            continue
        message = DiagnosticArray()
        message.header.stamp.sec = report["header"]["stamp"]["sec"]
        message.header.stamp.nanosec = report["header"]["stamp"]["nanosec"]
        message.header.frame_id = report["header"]["frame_id"]
        message.status = [
            DiagnosticStatus(
                level=bytes([status["level"]]), name=status["name"],
                message=status["message"],
                hardware_id=status["hardware_id"],
                values=[KeyValue(key=v["key"], value=v["value"])
                        for v in status["values"]])
            for status in report["status"]]
        messages.append(message)
    context = rclpy.context.Context()
    context.init(args=None)
    node = rclpy.create_node("acceptance_readiness_player", context=context)
    publisher = node.create_publisher(
        DiagnosticArray, "/easy_perception_deployment/detection_readiness",
        QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
    deadline = time.monotonic() + seconds
    index = 0
    while time.monotonic() < deadline and not stop.is_set():
        publisher.publish(messages[index % len(messages)])
        index += 1
        time.sleep(0.6)
    node.destroy_node()
    context.shutdown()
    return index


def scenario_script(acc, app, launcher):
    """Generator: yields milliseconds to wait between steps."""
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QApplication, QPushButton
    from windows.perception_readiness import ReplayState, load_session

    args = acc.args
    deploy = launcher.deploy_window
    monitor = deploy.perception_monitor
    video = sorted(str(p) for p in Path("/dev").glob("video*"))
    acc.check("1 launch without camera", launcher.isVisible(),
              f"launcher visible; /dev/video* = {video or 'none'}")
    acc.shot(launcher, "00_launcher")

    launcher.deployPackage()
    deploy.resize(QSize(args.width, args.height))
    yield 1500
    acc.check("2 open Deploy", deploy.isVisible() and
              monitor.mode == monitor.MODE_LIVE,
              f"Deploy {deploy.width()}x{deploy.height()}, live mode")
    acc.check("9 missing readiness (live)",
              monitor.table_stack.currentWidget() is monitor.unavailable_label
              and "READINESS UNAVAILABLE" in monitor.unavailable_label.text()
              and monitor.table_model.rowCount() == 0,
              f"badge {monitor.health_badge.text()}; "
              f"{monitor.unavailable_label.text().splitlines()[0]}")
    acc.shot(deploy, "01_deploy_workspace_live")

    # ---- recorded replay ------------------------------------------------
    session = load_session(args.session)
    monitor.replay_button.click()
    opened = monitor.open_session(args.session)
    acc.check("2 start replay workflow", opened and
              monitor.view_stack.currentIndex() == 1,
              f"opened {session.name}: {len(session.frames)} frames, "
              f"provenance {session.kind}")
    monitor.play()
    yield 2300
    index = monitor.player.index
    frame = session.frames[index]
    acc.check("3 replayed frames displayed",
              monitor.player.state == ReplayState.PLAYING and index >= 2
              and monitor.view._image is not None and
              monitor.view._image.width() == frame.image_width,
              f"frame {index + 1}/{len(session.frames)} shown at "
              f"{monitor.view._image.width()}x{monitor.view._image.height()}"
              f", badge {monitor.health_badge.text()}")
    acc.check("not shown as a live camera",
              "NOT A LIVE CAMERA" in monitor.provenance_label.text() and
              monitor.health_badge.text() == "REPLAY ACTIVE",
              monitor.provenance_label.text())
    acc.shot(deploy, "02_replay_running")

    monitor.pause()
    yield 200
    mismatches = []
    checked = 0
    for position in range(len(session.frames)):
        while monitor.player.index != position:
            monitor.step(1 if monitor.player.index < position else -1)
        expected = session.frames[position].readiness
        for row, record in enumerate(expected.records):
            cells = [monitor.table_model.text(row, c) for c in range(8)]
            want = [str(record.index), record.class_name,
                    f"{record.confidence:.2f}", record.track_id or "—",
                    record.state]
            if cells[:5] != want:
                mismatches.append((position, row, cells[:5], want))
            if (monitor.table_model.tooltip(row, 7) !=
                    f"{expected.stamp_ns // 10**9}."
                    f"{expected.stamp_ns % 10**9:09d}"):
                mismatches.append((position, row, "stamp"))
            checked += 1
        if not monitor.view.draws_rois():
            mismatches.append((position, "rois not drawn"))
    acc.check("4/7 detections, IDs, stamps preserved", not mismatches,
              f"{checked} table rows across {len(session.frames)} frames "
              f"match the recorded reports; mismatches={mismatches[:3]}")

    # Inspect a grasp candidate while paused.
    while monitor.player.index != 0:
        monitor.step(-1)
    row = next(i for i, r in enumerate(monitor.current_records)
               if r.state == "GRASP_CANDIDATE")
    monitor.select_row(row)
    yield 300
    record = monitor.current_records[row]
    values = monitor.inspector.values
    acc.check("6 inspect while paused",
              monitor.player.state == ReplayState.PAUSED and
              values["association"].text().startswith("Matched") and
              "GRASP_CANDIDATE" in values["readiness"].text() and
              values["dimensions"].text().startswith("Unavailable") and
              "not checked here" in values["downstream"].text(),
              f"#{record.index} {record.class_name} track "
              f"{record.track_id}; {monitor.inspector.usable.text()}")
    acc.shot(deploy, "03_inspector_selected")

    target = next(i for i, f in enumerate(session.frames)
                  if f.readiness and any(r.state == "REJECTED"
                                         for r in f.readiness.records))
    while monitor.player.index != target:
        monitor.step(1)
    row = next(i for i, r in enumerate(monitor.current_records)
               if r.state == "REJECTED")
    monitor.select_row(row)
    yield 300
    record = monitor.current_records[row]
    acc.check("6 rejected detection reason",
              "overlapping_ownership" in values["reasons"].text() and
              "not usable" in monitor.inspector.usable.text(),
              f"#{record.index} {record.class_name} track "
              f"{record.track_id}: {', '.join(record.reasons)}")
    acc.shot(deploy, "04_rejected_reason")
    localised = [r for f in session.frames if f.readiness
                 for r in f.readiness.records if r.state == "LOCALISED"]
    acc.check("recorded LOCALISED evidence", bool(localised) and all(
        r.centroid_m and r.axis is None for r in localised),
        f"{len(localised)} LOCALISED records with centroid, no axis")

    # Transport: restart, run to the end, stop.
    monitor.restart()
    yield 300
    restarted = monitor.player.index == 0 and (
        monitor.player.state == ReplayState.PLAYING)
    deadline = time.monotonic() + monitor.player.duration_s + 5
    while (monitor.player.state != ReplayState.ENDED and
           time.monotonic() < deadline):
        yield 250
    ended = monitor.player.state == ReplayState.ENDED
    acc.check("10 end of replay marked stale", ended and
              monitor.health_badge.text() == "STALE" and monitor.view._stale,
              f"badge {monitor.health_badge.text()} after "
              f"{monitor.player.duration_s:.1f} s")
    acc.shot(deploy, "05_replay_ended_stale")
    monitor.stop()
    yield 200
    acc.check("11 play/pause/restart/stop", restarted and
              monitor.player.state == ReplayState.STOPPED and
              monitor.table_model.rowCount() == 0 and
              not monitor.replay_timer.isActive(),
              "restart→frame 1 playing; stop→cleared, timer idle")

    # ---- synthetic fixture: all states and failure cases -----------------
    synthetic = FIXTURES / "synthetic_readiness_states"
    monitor.open_session(str(synthetic))
    monitor.step(1)
    yield 300
    states = [monitor.table_model.text(r, 4)
              for r in range(monitor.table_model.rowCount())]
    acc.check("5 all five readiness states (synthetic fixture)",
              sorted(states) == sorted(["REJECTED", "DETECTED", "LOCALISED",
                                        "POSE_VALID", "GRASP_CANDIDATE"])
              and "SYNTHETIC" in monitor.provenance_label.text(),
              ", ".join(states))
    monitor.select_row(2)
    yield 200
    acc.shot(deploy, "06_synthetic_all_states")
    monitor.step(1)
    monitor.step(1)
    yield 200
    acc.check("9 missing readiness (recorded frame)",
              "READINESS UNAVAILABLE" in monitor.unavailable_label.text() and
              monitor.table_stack.currentWidget() is
              monitor.unavailable_label,
              monitor.unavailable_label.text().replace("\n", " "))
    monitor.step(1)
    yield 200
    acc.check("8 mismatched association rejected",
              not monitor.view.draws_rois() and
              "do not belong together" in monitor.view._notice,
              monitor.view._notice)
    acc.shot(deploy, "07_mismatch_rejected")
    monitor.step(1)
    yield 200
    acc.check("10 missing RGB stream",
              monitor.view._image is None and
              "RGB frame missing" in monitor.view._placeholder,
              monitor.view._placeholder)
    acc.shot(deploy, "08_missing_rgb")

    # ---- production ROS topic path (optional, isolated domain) ----------
    ros = importlib.util.find_spec("rclpy") is not None
    if ros and not args.skip_ros:
        monitor.live_button.click()
        yield 500
        stop = threading.Event()
        published = []
        thread = threading.Thread(
            target=lambda: published.append(_publish_recorded_reports(
                args.session, 4.0, stop)), daemon=True)
        thread.start()
        deadline = time.monotonic() + 8
        while (monitor.health_badge.text() != "LIVE" and
               time.monotonic() < deadline):
            yield 200
        live_rows = monitor.table_model.rowCount()
        acc.check("ROS topic consumer (recorded messages)",
                  monitor.health_badge.text() == "LIVE" and live_rows > 0,
                  f"badge {monitor.health_badge.text()}, {live_rows} rows "
                  f"from {READINESS}; {monitor.health_state.messages}")
        acc.shot(deploy, "09_ros_topic_live")
        thread.join(timeout=10)
        deadline = time.monotonic() + 12
        while (monitor.health_badge.text() != "STALE" and
               time.monotonic() < deadline):
            yield 250
        acc.check("10 stale live output",
                  monitor.health_badge.text() == "STALE" and
                  monitor.summary_label.text().startswith("STALE"),
                  f"{published[0] if published else 0} messages published; "
                  f"then badge {monitor.health_badge.text()}: "
                  f"{monitor.health_state.messages}")
        acc.shot(deploy, "10_ros_topic_stale")
    else:
        acc.results.append({"scenario": "ROS topic consumer",
                            "status": "SKIPPED",
                            "detail": "rclpy unavailable or --skip-ros"})

    # ---- repeated start/stop cycles ---------------------------------------
    before = threading.active_count()
    for _cycle in range(args.cycles):
        monitor.replay_button.click()
        monitor.open_session(args.session)
        monitor.play()
        yield 120
        monitor.pause()
        monitor.select_row(0)
        monitor.stop()
        monitor.live_button.click()
        yield 60
    monitor.replay_button.click()
    yield 500
    after = threading.active_count()
    names = sorted(t.name for t in threading.enumerate())
    acc.check("12 repeated replay start/stop", after <= before and
              not monitor.live_subscribed() and
              not monitor.replay_timer.isActive() and
              len(monitor.cache) <= monitor.cache._capacity,
              f"{args.cycles} cycles; threads {before}→{after} {names}; "
              f"live subscription in replay: {monitor.live_subscribed()}")

    # ---- existing Deploy controls -----------------------------------------
    buttons = {b.text(): b for b in deploy.findChildren(QPushButton)}
    opened = []
    for text, title in (("Profiles && Replay", "EPD Profiles & Replay"),
                        ("3D Inspector", "EPD 3D Perception Inspector")):
        buttons[text].click()
        yield 400
        dialog = [t for t in QApplication.topLevelWidgets()
                  if t.windowTitle() == title and t.isVisible()]
        opened.append((text, bool(dialog)))
        for item in dialog:
            item.close()
    monitor.side_tabs.setCurrentWidget(monitor.setup_scroll)
    yield 300
    controls = (deploy.model_button, deploy.list_button,
                deploy.topic_button, deploy.usecase_config_button,
                deploy.confidence_spinbox)
    acc.check("13 Deploy controls and Profiles/Replay operational",
              all(ok for _t, ok in opened) and all(
                  monitor.setup_scroll.isAncestorOf(w) for w in controls)
              and deploy.run_button.isVisible(),
              f"dialogs {opened}; Run enabled={deploy.run_button.isEnabled()}"
              f" ({deploy.validation_label.text()})")
    acc.shot(deploy, "11_setup_tab")

    # ---- layout --------------------------------------------------------------
    monitor.open_session(args.session)
    monitor.play()
    yield 600
    monitor.pause()
    clipped = []
    for widget in (monitor.live_button, monitor.replay_button,
                   monitor.open_button, monitor.play_button,
                   monitor.pause_button, monitor.restart_button,
                   monitor.stop_button, monitor.health_badge,
                   deploy.run_button):
        right = widget.mapTo(deploy, widget.rect().topRight()).x()
        if (not widget.isVisible() or right > deploy.width() or
                widget.width() < widget.minimumSizeHint().width()):
            clipped.append(widget.objectName() or widget.text())
    acc.check(f"14 layout {deploy.width()}x{deploy.height()}",
              not clipped and monitor.view.height() >= 180 and
              monitor.table.viewport().height() >= 72,
              f"view {monitor.view.width()}x{monitor.view.height()}, table "
              f"viewport {monitor.table.viewport().height()} px, "
              f"clipped={clipped}")
    acc.shot(deploy, f"12_layout_{deploy.width()}x{deploy.height()}")
    deploy.resize(QSize(1920, 1080))
    yield 800
    acc.shot(deploy, f"13_layout_{deploy.width()}x{deploy.height()}")
    deploy.resize(QSize(args.width, args.height))
    yield 300

    deploy.close()
    yield 300
    acc.check("15 Deploy close stops workers",
              not monitor.workers_alive() and
              not monitor.live_timer.isActive() and
              not monitor.replay_timer.isActive(),
              f"readiness worker alive={monitor.workers_alive()}")
    acc.thread_names_before_exit = sorted(
        t.name for t in threading.enumerate())
    acc.write({"thread_names_before_exit": acc.thread_names_before_exit})
    for top in QApplication.topLevelWidgets():
        if top.isVisible():
            top.close()
    launcher.close()
    yield 100
    app.quit()


READINESS = "/easy_perception_deployment/detection_readiness"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=1366)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--skip-ros", action="store_true")
    args = parser.parse_args()
    acc = Acceptance(args)

    sys.path.insert(0, str(GUI_DIR))
    os.chdir(GUI_DIR)
    sys.argv = [sys.argv[0]]
    import main as epd_main
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    state = {}

    def advance():
        try:
            delay = next(state["script"])
        except StopIteration:
            return
        except Exception:  # noqa: BLE001 - recorded as a failure
            acc.check("harness", False, traceback.format_exc())
            acc.write({})
            QApplication.instance().quit()
            return
        QTimer.singleShot(delay, advance)

    def begin():
        launcher = next(
            t for t in QApplication.topLevelWidgets()
            if t.windowTitle() == "Easy Perception Deployment")
        state["script"] = scenario_script(acc, QApplication.instance(),
                                          launcher)
        advance()

    QTimer.singleShot(2500, begin)
    try:
        epd_main.main()
    except SystemExit:
        pass
    alive = [t.name for t in threading.enumerate()
             if t is not threading.main_thread() and not t.daemon]
    acc.check("15 application exit", not alive,
              f"event loop returned; non-daemon threads alive: {alive}")
    acc.write({"thread_names_before_exit":
               getattr(acc, "thread_names_before_exit", [])})
    print("HARNESS", "FAILED" if acc.failed else "COMPLETED", flush=True)
    return 1 if acc.failed else 0


if __name__ == "__main__":
    sys.exit(main())
