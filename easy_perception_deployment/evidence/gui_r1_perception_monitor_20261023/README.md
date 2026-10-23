# EPD GUI R1 — Perception Monitor, hardware-free acceptance (23 October 2026)

Branch `claude/epd-gui-r1-perception-monitor`, stacked on PR #235 (`codex/epd-r3-inference-performance` @ `355a4b9`, which contains the #234 readiness contract). No camera, GPU, Gazebo, robot or inference was used. Ubuntu 22.04, ROS 2 Humble, Python 3.10.

## Data

**Recorded (real).** The 12 sealed D435i RGB frames from 16 October 2026 (scenes A separated, B close/contact, C partial overlap; 1280×720) and the `detection_readiness` messages that the PR #234 hybrid-tracking node actually published for them were written byte-for-byte into a rosbag2. That bag was then exported with the new `epd_perception_session_export.py`. The bag's receive times are reconstructed; the GUI paces and pairs on header stamps only.

- RGB inputs were checked against their `capture.json` SHA256 seals before writing.
- The export check passed 12/12 on each count:
  - readiness dictionaries equal the original CDR messages;
  - PNG pixels equal the raw `rgb8` data;
  - image/report stamps and frames match.
- States: 28 `GRASP_CANDIDATE`, 4 `LOCALISED`, 8 `REJECTED` (`overlapping_ownership`). These match PR #234's reported replay.
- The data stays outside Git (17 MB per session) at `/home/user/epd-v2-artifacts/gui-r1-perception-monitor-20261023/`, with these seals:
  - `session_hybrid_tracking/session.json`: `e05cb9299da0292a01078a4abfe570476fa8c5403d55dcee1fac03730746c14e`
  - `bag_hybrid_tracking_0.db3`: `3324946359359fb6155c9836e6bfeaebdf32bcb5eedae8a1904359b2a4e0fe71`
  - `build_bags.py`: `d03a1664f767f7f81112090d640c63eafda228bd201fc5172cb7fe74f27ebb87`

**Synthetic (test fixture, not measurements).** `fixtures/perception_sessions/` provides drawn shapes and hand-written reports in the producer's exact format. They cover `DETECTED` and `POSE_VALID`, which the recording does not contain, plus a missing report, a mismatched pairing, a missing RGB frame, an empty report and a recording without the readiness topic.

## Real-application acceptance

`gui/perception_monitor_acceptance.py` starts the real `gui/main.py` with every layer on an X display (1366×768). ROS is sourced on an isolated domain. The process exit code is the verdict.

- Release code: **5/5 runs exit 0, 23/23 checks each** (`results.json` is run 1).
- Earlier runs on the final ROS and table design: 10/10.
- Stress: 750 Live↔Replay cycles, plus 1,350 on the previous revision.

| # | Scenario | Result |
|---|---|---|
| 1 | Launch with no camera (`/dev/video*` none) | PASS |
| 2 | Open Deploy, start Replay | PASS |
| 3 | Replayed 1280×720 frames shown | PASS |
| 4/7 | 40 rows across 12 frames match the recorded reports (index, class, confidence, track, state, stamp) | PASS |
| 5 | All five states (synthetic fixture) | PASS |
| 6 | Inspect while paused; rejected banana track 7 shows `overlapping_ownership` | PASS |
| 8 | Mismatched image/report stamps: no boxes drawn, reason shown | PASS |
| 9 | Readiness missing (live, frame, whole recording): READINESS UNAVAILABLE, not an empty table | PASS |
| 10 | End of replay STALE; live topic stale after publisher stops; missing RGB frame | PASS |
| 11 | Play/Pause/Restart/Stop | PASS |
| 12 | 20 start/stop cycles: no thread growth, no live subscription in replay | PASS |
| 13 | Profiles & Replay and 3D Inspector open; setup controls intact | PASS |
| 14 | 1366×768 layout: nothing clipped, view ≥ 180 px, table ≥ 3 rows | PASS |
| 15 | Deploy close stops workers; application exits with no non-daemon threads | PASS |

ROS-topic check: the recorded readiness messages were published on `/easy_perception_deployment/detection_readiness` in an isolated domain. The GUI showed LIVE, noting that the publisher was not started from this window. It turned STALE 5 s after the publisher stopped.

Screenshots were grabbed from the running windows:

| File | Shows |
|---|---|
| `01` | Workspace (live, readiness unavailable) |
| `02` | Replay running |
| `03` | Inspector on a grasp candidate |
| `04` | Rejected detection and its reason |
| `05` | Replay ended (stale) |
| `06` | Synthetic all-states fixture |
| `07` | Mismatch rejected |
| `08` | Missing RGB frame |
| `09`, `10` | ROS topic live, then stale |
| `13` | Large layout (1848×1013, the display maximum) |

## Tests

- **GUI CI step with CI's pins** (PySide6 6.11.2, pytest 7.4.4, no ROS):
  - this branch: 132 passed, 2 skipped (ROS-only), 5/5 runs;
  - base: 107 passed.
  - CI lint passes: strict and baseline pycodestyle, ruff 0.17.1.
- **System environment** (PySide6 6.11.0), 4 runs each mode:
  - without ROS: 143 passed, 2 skipped;
  - with ROS: 145 passed.
  - In both modes the only failures are the same 4 that fail on base: `test_runLabelme`, `test_deploy_import_path` and two `test_fps_monitor` tests. None of them is in GUI CI.
- `test_perception_monitor.py`: 27 tests. These include the rosbag2 exporter round trip and the live `ReadinessTopicMonitor` on an isolated ROS domain.

## Defects found during acceptance

- **PySide6 6.12.0 on Python 3.10** drops a reference to `None` on every void Qt call and aborts with `none_dealloc`. The unmodified base app aborted 2/2 in a soak. The fix pins GUI CI to 6.11.2 and excludes 6.12.0 in `requirements.txt`.
- **Per-frame `QTableWidgetItem` replacement** crashed in `Shiboken::BindingManager::releaseWrapper` after certain earlier tests. The fix is model/view (`QAbstractTableModel`).
- **Existing Run button:** the legacy job-state handler reset its icon to 100 px, which clipped it. Restored.
- **Inspector:** wrapped values were clipped when a scrollbar appeared. Fixed.
- **ROS context churn:** switching modes created and destroyed a ROS context each time. The context now persists while Deploy is open, and only the subscription toggles.
- **Ghost Train/Deploy windows (pre-existing in #235; fixed in the follow-up commit):** with ROS sourced, the base GUI pytest suite segfaults in roughly 1 in 3 runs, in PySide deferred wrapper deletion. See "GUI CI segfault" below.

**Open:** one SIGSEGV in a real-app run, on an earlier revision, during the transition into the start/stop cycles. It did not reproduce in 36 later full runs (plus 6 under gdb) or 2,100 stress cycles. Its cause is unknown.

## GUI CI segfault (follow-up commit)

GitHub GUI CI failed twice with SIGSEGV (exit 139) inside existing `test_gui.py` tests, after `QObject::killTimer: Timers cannot be stopped from another thread`.

**Root cause, proven.** `MainWindow` creates its Train and Deploy windows as parentless top-level widgets that only Python attributes hold. After a test deletes `MainWindow`, both windows stay alive in a Python reference cycle, still running their UI timers. Nothing destroys them except Python's cyclic GC. That can run on any thread that executes Python, such as Deploy's topic-discovery `QThread` or ROS threads, so Qt objects are torn down off the GUI thread.

PySide6 6.12.0 defers such deletions to the main thread and masked the problem; #240's pin to 6.11.2 exposed it.

**Reproducer.** Run the CI test list with CI's pins, and after each test run `gc.collect()` on a worker thread:

| Tree | Result |
|---|---|
| Base #235 | 3/3 segfault, with `killTimer` |
| #240 before the fix | 3/3 segfault |
| #240 with the fix | 6/6 clean, 132 passed, 2 skipped |

A forced main-thread GC after each test segfaulted 4/4 on base and is clean with the fix.

**Fix.** `MainWindow` deletes its Train and Deploy windows on the GUI thread when it is destroyed, via `destroyed` → `deleteLater`. The monitor also stops its ROS worker when its controller is destroyed. A new `test_gui.py` regression test fails without the fix.

**After the fix:**
- Exact CI command with CI pins: 10/10 clean, 133 passed, 2 skipped, no cross-thread timer warnings.
- Forced off-thread GC, 5 runs: no cross-thread destruction.
- ROS-sourced workstation runs: 0/8 crashed (base: 3/8).
- Real-app acceptance: 3/3 runs exit 0, 23/23 checks each.

## Not covered

Live camera operation, GPU performance, perception accuracy on new scenes, Stage-A2 and robot execution.
