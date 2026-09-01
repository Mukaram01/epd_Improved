# EPD Production Hardening Roadmap

Baseline: ROS 2 Humble on Ubuntu 22.04. EPD owns camera ingestion, perception,
localization, tracking, and perception health. EMD owns planning and execution.
Robot motion remains disabled by default.

## Current evidence

- RealSense D435i RGB runs at approximately 14.98 Hz.
- During the reproduced EPD stall, the camera publisher and EPD subscription
  endpoint remain present and QoS-compatible.
- An independent 15-second probe received RGB=224, depth=223, CameraInfo=224,
  with 222 exact timestamp triplets.
- In the stalled EPD process, RGB callback entry and exit both stopped at 1014,
  while depth and CameraInfo callbacks continued. Dispatch entry and exit both
  stopped at 997.
- Therefore no EPD callback, matcher, or localization dispatch was blocked. The
  active EPD process stopped receiving RGB samples internally before callback
  entry. ApproximateTime tuning is not a supported P1 direction.

## Phase gates

| Phase | Scope | Dependencies | Hard acceptance gate | Status |
|---|---|---|---|---|
| P0 | Baseline audit and reproducible acceptance harness | None | Versioned harness records topic counts, first/last sensor stamps, observation/output freshness, process RSS, and test duration | Complete |
| P1 | Isolated, bounded sensor ingress | P0 | 640x480x15 soak for at least 30 minutes; RGB/depth/info and observation generation advance for full run; bounded RSS; EPD restart and camera reconnect tested; start/end counts recorded | ACCEPTED; physical USB unplug subtest blocked |
| P2 | Immutable synchronized Observation with `observation_id` | P1 | One immutable RGB/depth/intrinsics/TF-quality unit; timestamp truth; compatibility adapter for existing EPD interfaces | PASS |
| P3 | Inference/backpressure separation | P2 | Latest-only bounded queue; ingress remains live under deliberately slow inference | PASS |
| P4 | Localization geometry and quality truth | P2-P3 | Calibrated frame semantics, depth validity, pose/dimension quality flags, deterministic fixtures | PASS |
| P5 | Tracking and object lifecycle | P2-P4 | Stable IDs and explicit appeared/updated/lost lifecycle under replay | PASS |
| P6 | Streaming plus fresh-service semantics | P2-P5 | Baseline N can only accept completed observation_id>N; timeout is safe; no stale replay | PASS |
| P7 | EPD-EMD handshake | P6 | EMD consumes fresh snapshots without pausing EPD ingress; compatibility preserved | PASS |
| P8 | Record/replay and fixtures | P2-P7 | Deterministic fixture replay covers production contracts without camera hardware | PASS |
| P9 | Health, metrics, recovery | P1-P8 | Explicit healthy/degraded/stalled/recovering states and tested lifecycle recovery | Blocked |
| P10 | Model-engine modernization | P3, P8-P9 | Backend-neutral results and benchmarked optional acceleration | Deferred |
| P11 | Calibration, TF, depth quality | P4, P8 | Calibration provenance and TF/depth fault coverage | Deferred |
| P12 | Camera profiles and multi-camera readiness | P2, P9, P11 | Named camera identity and independent bounded ingress instances | Deferred |
| P13 | Workcell Studio profiles | P9-P12 | Generated validated perception profiles with no task logic in EPD | Deferred |
| P14 | Existing EMD grasp integration | P7-P13 | Fresh perception snapshot drives grasp planning in fake-hardware mode | Deferred |
| P15 | PlanningScene and reachability | P14 | Collision/reachability checks consume truthful EPD geometry | Deferred |
| P16 | Fake-hardware perception-to-pick/place | P15 | End-to-end deterministic test with motion disabled unless explicitly enabled | Deferred |
| P17 | Fault injection and endurance | P1-P16 | Camera/network/process/model faults plus long-duration soak pass | Deferred |
| P18 | Gazebo, Isaac, advanced models | P17 | Added only after transport, freshness, and health gates remain green | Deferred |

## P0/P1 implementation direction

The camera-facing DDS readers move into a lightweight `epd_sensor_ingress`
process. It performs no inference or task work and immediately republishes each
message with its original header and payload onto bounded internal topics.
Camera-facing subscriptions use explicit RELIABLE KEEP_LAST(1), matching the
observed RealSense publisher. Internal publishers and EPD subscribers use
BEST_EFFORT KEEP_LAST(1). Process separation gives ingress independent DDS
reader resources, executor ownership, failure domain, and restartability.

The isolated ingress passed the P1 runtime gate below. The bounded in-process
exact timestamp matcher now publishes immutable Observations into a single-slot
latest store. Temporary callback entry/exit instrumentation was removed after
acceptance; health, stall, drop, inference, and Observation progress diagnostics
remain.

## Evidence log

- Focused freshness/ApproximateTime regression suite: passed before P1 work.
- Package build with `epd_sensor_ingress`: passed.
- Synthetic isolated-ingress exercise: 10 seconds at 15 Hz; publisher sent 148
  synchronized triplets; ingress counters advanced RGB/depth/info from
  296/296/296 to 444/444/444; original stamps were preserved across raw and
  internal topics; observed ingress RSS peak was 14,072 KiB; clean SIGINT
  shutdown passed. This is functional evidence only, not the P1 endurance gate.
- Health diagnostics correctly transitioned to stale after the synthetic camera
  stopped and reported counts, receive ages, and last sensor stamps.
- Real D435i soak: 1,805.10 seconds at configured 640x480x15. Authoritative
  ingress callbacks advanced RGB 262->19,726 (10.783 Hz), depth 307->12,086
  (6.526 Hz), and CameraInfo 284->15,737 (8.561 Hz). Drops were bounded and
  streams remained live at the end.
- Synchronized observation generation advanced 117->6,334 (3.441 Hz). Two
  transient gaps exceeded five seconds and recovered automatically; permanent
  stall count was zero.
- Throttled inference diagnostics emitted 634 completion samples during the
  soak (0.351 Hz lower bound); reported instantaneous inference was typically
  0.45-0.75 FPS. Probe-observed localization and image outputs remained fresh at completion,
  with ages 2.92 seconds and 0.86 seconds respectively.
- RSS KiB start/peak/end: ingress 23,660/26,296/25,000; EPD
  1,305,932/1,414,172/1,347,144. No unbounded growth was observed.
- Hardware timestamp audit: 15.13 seconds; 219 exact raw triplets retained;
  source-to-ingress stamp matches RGB=116, depth=103, CameraInfo=63; duplicate
  and regressed stamp counts were zero for raw, ingress, localization, and image
  output streams.
- EPD/ingress restart with RealSense continuously live passed. Pre-restart output
  stamp 1788131795.002899414; post-restart localization stamp
  1788131847.309019287. Generation restarted cleanly and no previous-process
  result was replayed.
- RealSense software restart passed. Ingress health changed to ERROR with stale
  ages while stopped, returned to OK automatically after restart, and both EPD
  outputs resumed with the same fresh sensor stamp 1788131921.001571533.
- Continuous-mode service lifecycle bug fixed: requests now offer fresh
  generations even when streaming inference is enabled. Request 1 baseline 128,
  accepted 141, success=true, stamp 1788132104.536189209. Request 2 baseline
  404, accepted 420, success=true, stamp 1788132131.686791992.
- Physical USB unplug/replug was not performed because it requires a physical
  action. Only that subtest remains blocked.

## P2 immutable Observation acceptance

`Observation` is the sensor-truth contract for one synchronized input. It owns
shared immutable RGB, aligned depth, and CameraInfo messages plus the original
RGB sensor stamp/frame, camera identifier, synchronization quality, source
health, and optional TF status. `LatestObservationStore` assigns the sole
process-local monotonic `observation_id`, retains one latest Observation, and
provides thread-safe `latest`, `latest_after`, and blocking `wait_for_newer`
operations with shutdown notification. Service state and perception results do
not mutate or live inside an Observation.

- The localization/tracking callbacks only construct/publish Observations and
  wake the existing worker. The worker holds shared immutable input ownership;
  output adapters preserve the source header. The fresh-service baseline and
  completed-result identity both use `observation_id`; the parallel generation
  helper/counters were removed.
- Deterministic Observation regression suite: 9/9 passed, covering immutable
  ownership and sensor headers, strictly increasing/process-local IDs,
  latest-only bounded retention, strict `latest_after`, stale timeout,
  wait/wakeup, consecutive request baselines, shutdown, and concurrent access.
  The relevant package build passed. Launch Python validation and
  `git diff --check` passed.
- Real D435i regression: 75.151 seconds using the production `run.launch.py`
  path. Observation diagnostics advanced 0->1 and continued through at least
  579 with advancing source stamps and no permanent stall.
- Probe counts over that interval: ingress RGB=725, depth=462,
  CameraInfo=709; localization=16 and image output=14. Every stream had zero
  duplicate and zero regressed stamps. End ages were 0.018 s RGB, 0.107 s
  depth, 0.017 s CameraInfo, 1.294 s localization, and 1.300 s image output.
- Service request 1 accepted observation 457 above baseline 454,
  `success=true`, source stamp 1788167272.331010742. Request 2 accepted
  observation 512 above baseline 506, `success=true`, source stamp
  1788167300.485082275. Neither request reused its predecessor.
- RSS KiB start/peak/end: ingress 21,972/25,168/23,928; EPD
  1,312,896/1,401,648/1,337,320. The store retained exactly one latest
  Observation in deterministic stress coverage and runtime RSS remained
  bounded.
- The full legacy test invocation also reported existing repository-wide lint
  debt and two model-content assertions with empty detections; the focused P2
  regression and integration build passed. Those unrelated failures do not
  invalidate Observation freshness or the hardware result.

## P3 backpressure-proof inference acceptance

The camera-facing process, exact matcher, and `LatestObservationStore` remain the
P1/P2 path. Observation callbacks only create immutable Observations and submit
their process-local `observation_id` to `LatestInferenceScheduler`. The
scheduler owns one latest-only slot and one condition-variable-driven worker;
only that worker executes ORT, PCL, geometry, and visualization. While it is
busy, newer Observations replace the pending slot and displaced IDs are counted
as intentionally skipped. Localization, continuous mode, and service requests
share this worker and `PerceptionResult(source_observation_id, sensor_stamp,
frame_id)`. There is no service-only inference path or second freshness counter.

- Deterministic scheduler suite: 13/13 passed, including fast-producer/slow-
  consumer bounding, newest selection, monotonic/nonduplicate consumption,
  exact skip accounting, sleep/wakeup, shutdown, service baselines and
  consecutive requests, continuous-result reuse, failure recovery, 10,000-item
  bounded stress, and source stamp/frame preservation. Together with the P2
  Observation suite, 22/22 focused cases passed.
- Real D435i regression: 300.167 seconds through production `run.launch.py`.
  Ingress probe counts were RGB=2,991 (9.965 Hz), depth=1,517 (5.054 Hz), and
  CameraInfo=2,772 (9.235 Hz). Scheduler Observations advanced 84->1,122
  (1,038 produced, 3.458 Hz over the measured interval) while inference was
  deliberately slower.
- Inference advanced started 11->113 and completed 10->112; failed remained
  zero. Completion rate was 0.340 Hz over the interval. Latency min/average/max
  was 2,088/2,805/5,772 ms at completion. Consumed ID advanced 79->1,117 and
  completed ID 74->1,102. Intentional skips advanced 68->1,004 (936). Backlog
  high-water mark was one; duplicate/regressed submissions and duplicate
  processing attempts were zero.
- End ages were 396 ms for the newest Observation and 5,266 ms for the newest
  completed result. Probe output stamps had zero duplicates/regressions and
  remained original sensor stamps. Two transient exact-sync warnings recovered;
  permanent stalls and stale replays were zero.
- RSS KiB start/peak/end: ingress 24,372/24,416/22,880; EPD
  1,380,380/1,461,436/1,400,112. No unbounded growth or full-frame backlog was
  observed.
- Continuous-mode services shared the single worker. Request 1 baseline 1,230
  accepted 1,231, `success=true`, stamp 1788169297.813480713. Request 2 baseline
  1,681 accepted 1,685, `success=true`, stamp 1788169455.864006592. Neither
  accepted a cached result at or below its baseline.
- EPD/ingress restart with RealSense live passed: both processes exited cleanly,
  reconnected automatically, scheduler identity restarted at 1, and fresh
  inference resumed without old-process replay. RealSense software stop produced
  an honest idle/stale state at Observation 146; restart recovered automatically
  at Observation 147 and continued beyond 287 with fresh results.
- Machine-readable inference diagnostics publish at 1 Hz and retain the P1
  health diagnostics. The probe subscribes to both and exits cleanly. Relevant
  package build, launch validation, Python syntax checks, and `git diff --check`
  passed. Physical USB reconnect remains the accepted P1-only blocked subtest;
  it is not a P3 gate.

P3 is complete; its architecture remains the input to P4.

## P4 manipulation-grade perception truth acceptance

The audited localization path is MaskRCNN boxes/classes/scores/masks, aligned
depth association, masked point-cloud projection, PCA axis, metric geometry,
legacy `LocalizedObject` adaptation, publication, and service caching. RGB,
depth, intrinsics, mask, and geometry all derive from one immutable Observation;
the internal object carries its `observation_id`, original sensor stamp, and
optical frame. No TF is requested or applied in this path. Coordinates, cloud
points, centroid, and length/breadth/height are metres in the camera optical
frame; ROI and masks are pixels. The pose position is the validated centroid
and its normalized quaternion is derived deterministically from the validated
PCA major axis.

The audit found that the previous path could divide by invalid focal lengths,
index an out-of-image ROI, publish zero/non-finite dimensions, and substitute a
fabricated +Z axis for empty or degenerate point clouds. P4 adds an internal
`LocalizedObject` truth record with confidence, source identity, depth-support
counts/ratio, `VALID`/`DEGRADED`/`INVALID`, and a machine-readable failure
bitmask. Validation rejects invalid intrinsics, ROI/mask mismatch, insufficient
depth, empty/non-finite clouds, non-finite centroid, non-positive dimensions,
and non-finite/non-unit orientation. Invalid depth values are excluded. Empty
or degenerate clouds no longer acquire a valid-looking axis.

Thresholds are ROS parameters with conservative defaults: minimum 16 mask
pixels, 12 valid depth pixels, 0.20 valid-depth ratio, and 12 finite cloud
points. Insufficient depth alone is `DEGRADED`; any hard geometry violation is
`INVALID`. Only `VALID` objects enter the unchanged legacy `epd_msgs` object
array and pose output. Thus a healthy zero-detection frame remains a successful
empty result, while detections with unusable geometry are omitted rather than
converted to plausible zeros. Public message definitions and EMD compatibility
were not changed.

- Deterministic geometry-quality suite: 15/15 passed, covering masked-depth
  centroid projection, positive dimensions, NaN/Inf rejection, invalid
  intrinsics, empty mask/cloud, insufficient support, out-of-bounds ROI, invalid
  depth exclusion, unit orientation, explicit degeneracy, source identity,
  mixed-ID detection, per-object failure isolation, and healthy zero detections.
  With P2/P3 suites, 37/37 focused cases passed.
- Final Real D435i acceptance: 300.172 seconds through production
  `run.launch.py`. Observations advanced 108->1,212 (1,104), inference completed
  9->114 (105) with zero failures, and output stamps had zero duplicates or
  regressions. End Observation/result ages were 1,325/6,460 ms; probe-observed
  localization age was 759 ms.
- The scene produced zero model detections. Counters remained detections=0,
  valid=0, degraded=0, invalid=0 and every failure-reason counter zero. This is
  the intended healthy-empty policy. Live validation of an actual localized
  COCO object is blocked only by the current scene and is not a P4 gate because
  deterministic geometry fixtures passed.
- RSS KiB start/peak/end: ingress 24,012/25,416/22,700; EPD
  1,500,488/1,526,824/1,463,032. Backlog high-water remained one, stale replay
  count was zero, and duplicate/regressed IDs were zero.
- Production diagnostics expose detection and quality totals, failure-reason
  counters, and latest-valid-geometry age at 1 Hz without per-object logging.

P4 is complete.

### P4 live-acceptance clock regression correction

Live acceptance exposed an abort on the first valid geometry result. P4 stored
the source ROS sensor stamp as nanoseconds, then diagnostics reconstructed it
with `rclcpp::Time(int64_t)`, which defaults to `RCL_SYSTEM_TIME`. Subtracting
that value from `Node::now()` (`RCL_ROS_TIME`) threw `can't subtract times with
different time sources [1 != 2]` in `latest_valid_geometry_age_ms`.

Age calculation now constructs sensor times explicitly in the current node
clock domain. A no-throw helper checks clock types before arithmetic and reports
an unavailable age for mixed domains. Source stamps remain unchanged;
inference elapsed time continues to use `std::chrono::steady_clock`.

- Four clock tests passed: same-clock subtraction, safe mixed-clock handling,
  immutable sensor stamp, and finite/nonnegative valid age. All 41 focused
  P2-P4 cases passed; package build and `git diff --check` passed.
- Corrected real D435i run: 300.202 seconds. Observations advanced 453->1,447;
  inference completed 74->178 (104 completions), failed remained zero, and the
  process did not abort. Result-age diagnostics remained finite/nonnegative and
  ended at 3,992 ms. Stale replay and duplicate/regressed ID counts were zero.
- The live scene produced 104 detections and 104 VALID geometries during the
  measured interval, with zero degraded/invalid results. A sampled `mouse`
  result had ROI/mask 112x130, 8,994 segmented points, centroid
  (0.0476, -0.0626, 0.5171) m, dimensions
  (0.1132, 0.0610, 0.0390) m, and quaternion norm 1.00000004.

## P5 temporal identity acceptance

The legacy tracking audit found OpenCV image trackers with numeric string IDs,
fixed 2D thresholds, a three-frame miss limit, and an unbounded ID log. They
did not consume `observation_id`, geometry quality, or 3D position. Continuous
and service outputs shared the node, but temporal freshness was not part of the
tracker contract.

P5 retains the production Observation/inference/result path and adds one
deterministic tracker owner after P4 geometry validation. Process-local IDs are
monotonic and never recycled. Class-gated greedy association uses ROI IoU and
centroid distance, plus 3D centroid distance only for VALID geometry; stable
tie-breaking prevents iteration-order ambiguity. Defaults are IoU 0.20, ROI
distance 100 px, valid-3D distance 0.25 m, confirmation after two hits, three
missed Observations of grace, 64 active tracks, and velocity dt 0.001-5 s.
Tracks progress TENTATIVE -> CONFIRMED -> LOST and are removed as EXPIRED;
reappearance after expiry receives a new ID. Active storage is capped and no
per-track history is retained. Duplicate/out-of-order Observation updates are
rejected. Velocity is available only from two valid 3D positions and a sane
positive source-stamp delta. The legacy public message remains compatible and
uses the P5 IDs; machine-readable diagnostics expose lifecycle, association,
ordering, latency, age, and geometry-quality counters.

- Deterministic tracker suite: 16/16 passed, including stable and distinct IDs,
  miss/expiry/reappearance, ordering and duplicate rejection, class/jump gates,
  deterministic association, valid/invalid velocity, healthy empty frames,
  confidence separation, bounded stress, and malformed-object isolation. All 57 focused P1-P5 cases
  passed.
- Real D435i production run: 305.231 seconds. Observations advanced 581->1,676;
  inference completed 66->169 (103), failed remained zero, and detections
  advanced 66->168. One real `mouse` retained ID `1` across 51 probe-observed
  outputs. Tracks created/confirmed/expired remained 1/1/0; one transient miss
  entered LOST and then recovered the same ID. Active/max-active were 1/1;
  associations advanced 65->167 with zero rejections or detectable switches.
  Tracker duplicate/out-of-order updates, output stamp duplicates/regressions,
  and stale replay were zero. Tracker latency ended at 13 us.
- RSS KiB start/peak/end: ingress 23,036/26,056/24,320; EPD
  1,406,680/1,483,128/1,422,560. Service freshness passed twice on shared state:
  completed Observation 461->468 and 468->489, both success=true with strictly
  newer sensor stamps and track ID `1`.
- Slow movement, controlled occlusion/expiry, and two same-class live objects
  are blocked only by the static scene; deterministic fixtures cover them.

P5 is complete.

## P6 authoritative completed-result contract

The audit found one correct single inference worker and an existing bounded
`LatestPerceptionResultStore`, but the store held only source metadata. Service
responses were assembled from separate mutable localization/tracking caches,
a readiness flag, a second result ID, and a second condition variable. That
duplicated ownership left the payload-to-source relationship implicit.

P6 makes the existing store the sole completed-result timeline. Each immutable
depth-one result owns its source `observation_id`, original sensor stamp/frame,
success state, exact localization and tracking payloads, and steady-clock
completion time. The inference worker constructs and publishes it only after
the corresponding continuous payload is complete. Service requests serialize,
record the latest Observation ID as baseline, and block on the store until
`source_observation_id > baseline`; they copy both response fields from that
same immutable result. No service inference path, result queue, timestamp
rewrite, or tracker instance exists.

The store rejects and counts duplicate/regressed publishes, waits without busy
polling, retains one result, and wakes all waiters on shutdown. Timeout returns
`success=false` with empty response messages and does not mutate inference or
tracking state; a later request can succeed. A successful fresh result with
zero objects remains `success=true`. Diagnostics expose completed-result,
service, timeout/shutdown, baseline/result, age, ordering, duplicate, and waiter
counters.

- Ten focused result-store contract cases passed within the 19-case scheduler
  suite: strict baseline, successor acceptance, depth one, regression/duplicate
  rejection, timeout then recovery, shutdown wake, healthy zero detection,
  payload/source consistency, and consecutive freshness. All 63 focused P1-P6
  cases passed.
- Real D435i production measurement: 75.178 seconds. Observations advanced
  60->270 and inference completed 7->28 (21 completions), with zero failures,
  stale replay, submission/result regression, or duplicate publish. Completed
  result ID advanced 56->250 and continuous localization/image output remained
  live with truthful, non-regressed stamps.
- Consecutive services passed on shared continuous state: request 1 baseline
  395 accepted result 399, `success=true`, sensor stamp
  1788193252.832456299, one object; request 2 baseline 411 accepted result 423,
  `success=true`, stamp 1788193259.173050537, zero objects. The latter directly
  validates fresh healthy-empty semantics. No service timeout occurred in live
  operation; deterministic timeout/recovery coverage passed.

P6 is complete.

## P7 EPD-EMD ownership and freshness handshake

The audited integration keeps the existing public topics and service. EPD owns
the D435i ingress, immutable Observation generation, the latest-only inference
worker, authoritative completed results, localization/tracking publications,
and the fresh-result service. EMD subscribes to localization or tracking,
decides whether to plan, owns execution state, and may call
`epd_perception_service` as a freshness/watchdog request. There is no second
perception pipeline and neither process owns the other's work.

The P7 correction removes the remaining legacy service-mode scheduling gate:
every synchronized Observation is now submitted to EPD's worker whether or not
an EMD request is active. A service request only records baseline N and waits
for the authoritative result store to complete `source_observation_id > N`.
Consequently EMD busy state, disappearance, or restart cannot pause camera
ingress, Observation creation, inference, or continuous publications. Healthy
zero-detection results remain successful fresh perception.

EMD's existing `execution_in_progress_gate_enabled` gate suppresses grasp
planning/execution while busy. When
`pause_epd_triggers_while_execution_in_progress=true`, it additionally
suppresses only redundant service/watchdog trigger calls; it does not publish a
pause command and cannot affect EPD sensing or scheduling. An in-flight EPD
request older than `epd_msg_timeout_s` is abandoned locally so an EPD restart
cannot permanently suppress retries. EPD restart creates empty process-local
Observation/result stores and therefore cannot replay an old result; EMD
restart carries no EPD generation state and EPD continues uninterrupted.

- Focused deterministic acceptance passed: 14 result/freshness/P7 cases, nine
  Observation cases, 15 geometry cases, 16 tracking cases, four time cases,
  and three EMD trigger/restart cases. Coverage includes busy/absent consumer
  progression, configured trigger suppression, post-busy strict freshness,
  both restart directions, healthy empty results, shutdown wake without
  deadlock, and all P6 store semantics.
- Both `easy_perception_deployment` and `emd_grasp_planner` built successfully.
  Launch files passed Python compilation. The wider legacy EMD `grasptest`
  binary still has unrelated existing failures outside the focused P7 cases;
  these were not changed or counted as P7 failures.
- Real D435i acceptance ran 75.151 seconds at 640x480x15. Observations advanced
  48->264 and inference completed 4->32 with zero inference failures. Ingress
  RGB/depth/info, localization, and image output stayed live while the
  EMD-equivalent consumer was present and during a no-trigger busy interval.
  All monitored streams had zero duplicate and zero regressed stamps;
  scheduler and result-store duplicate/regression counters were zero.
- Fresh services succeeded with baseline/result 338/345, then after the client
  process restarted with 96104/96111. EPD continued from Observation 264 to
  96159 while that consumer was absent/restarted. After EPD restart its stores
  started empty, Observations restarted at 1 and advanced to 80, inference
  completed to 11, and a new service succeeded at baseline/result 46/48 with
  no result regression or duplicate publish.

P7 is complete. The following section records the current P8 increment.

## P8 deterministic record/replay

P8 adds a small, versioned JSON fixture and `epd_replay.py`, a sensor source
that replaces only physical camera ingress. It publishes ordered RGB, aligned
depth, and CameraInfo triplets with their recorded source stamp, frame, camera
identity, intrinsics, and exact-synchronization truth. The existing exact
matcher then creates Observations through `LatestObservationStore`; fixture
files never encode `observation_id`. Production ONNX inference, latest-only
backpressure, geometry quality, temporal tracking, completed-result storage,
fresh services, output topics, and P7 diagnostics are unchanged and are not
reimplemented by replay.

No RealSense is required. From a built and sourced workspace, run:

```bash
ros2 launch easy_perception_deployment replay.launch.py \
  mode:=fast summary_output:=/tmp/epd_replay_summary.json
```

`fast` preserves fixture order but does not pace against sensor time;
`realtime` is available for operator inspection. Playback stops at EOF and the
launch fails when acceptance fails. The JSON summary contains stable contract
evidence only, so two `fast` runs can be compared directly after excluding no
fields. The committed fixture plus focused production contract suites cover
appeared/updated/stable/lost tracking, duplicate and regressed source stamps,
invalid depth geometry, strict result-store counters, and the bounded
latest-only scheduler. Existing focused P6
tests remain the authoritative strict-baseline proof that a cached result at N
cannot satisfy a request requiring a completed `observation_id > N`.

Replay is development and CI infrastructure. It does not command robot motion,
does not contain task or planning logic, and does not replace physical-camera
acceptance. EMD/Workcell Studio remains a separate consumer of the same EPD
topics and fresh-result service.

P8.1 fixed the Mask R-CNN initialization failure. Model input inspection held
a borrowed `TensorTypeAndShapeInfo` after its temporary owning `TypeInfo` had
already been destroyed; the dangling view produced a bogus dimension count and
`std::vector` raised `cannot create std::vector larger than max_size()`. The
owner now remains alive through the shape query, and a fixture-to-Observation
regression verifies message byte/step/type/contiguity truth plus real Mask
R-CNN initialization and inference.

P8.2 replaces the non-detectable replay image with the repository's existing
model-detectable portrait and supplies internally consistent 320x240 camera
intrinsics and synthetic sensor-depth regions. It does not encode detections or
change the 0.50 production confidence threshold. Tracking mode can now select
the configured production tracker explicitly, and confirmed track IDs are
retained in production diagnostics so acceptance does not depend on delivery of
the large best-effort tracking message.

Two exact `fast` replay runs produced identical PASS summaries: eight accepted
and completed Observations, two appeared and lost tracks, four associations,
stable IDs 1 and 2, four valid and two invalid geometry results, two duplicate
or regressed source timestamps, zero stale results, and a backlog high-water
mark of one. P8 is complete as deterministic offline production-contract
coverage; this is not a physical-camera acceptance claim.
