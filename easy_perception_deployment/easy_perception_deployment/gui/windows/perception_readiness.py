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

"""Readiness reports, recorded sessions and replay timing for Deploy.

Pure Python: no Qt widgets and no ROS imports, so the GUI CI can test it.

The producer is ``EPD::makeReadinessReport`` (perception_readiness.hpp): one
``diagnostic_msgs/DiagnosticStatus`` per detector proposal on
``/easy_perception_deployment/detection_readiness``, stamped with the source
RGB header. This module only reads that contract. It never derives a
readiness state, a pose or a reason that the report did not contain.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path


READINESS_TOPIC = "/easy_perception_deployment/detection_readiness"
STATUS_PREFIX = "easy_perception_deployment/detection/"

STATES = (
    "REJECTED",
    "DETECTED",
    "LOCALISED",
    "POSE_VALID",
    "GRASP_CANDIDATE",
)

# Operator wording for the producer's state definitions.
STATE_TEXT = {
    "REJECTED": (
        "Rejected",
        "The 2D evidence is unusable or contested. Class and confidence "
        "are still reported so the object does not disappear.",
    ),
    "DETECTED": (
        "Detected, no 3D position",
        "Class, confidence and an owned 2D mask, but no validated 3D "
        "position.",
    ),
    "LOCALISED": (
        "Localised, no orientation",
        "The 3D position was measured from the mask, but the orientation "
        "is not valid.",
    ),
    "POSE_VALID": (
        "Pose valid, not published",
        "Position and orientation are valid, but EPD did not publish this "
        "pose (see the note).",
    ),
    "GRASP_CANDIDATE": (
        "Grasp candidate",
        "Published on EPD's valid-pose output for downstream planning.",
    ),
}

DOWNSTREAM_NOTICE = (
    "Perception readiness only. Reachability, collisions and grasp success "
    "are decided downstream and are not checked here."
)

# Failure names emitted by EPD::failureNames (geometry_quality.hpp).
REASON_TEXT = {
    "invalid_intrinsics": "Camera intrinsics are invalid.",
    "roi_out_of_bounds": "The detection box lies outside the image.",
    "empty_or_mismatched_mask": "The object mask is empty or mismatched.",
    "insufficient_depth": "Too few valid depth pixels inside the mask.",
    "too_few_segmented_points": "Too few 3D points after segmentation.",
    "nonfinite_geometry": "The 3D geometry is not finite.",
    "invalid_dimensions": "The measured dimensions are invalid.",
    "degenerate_axis": "Orientation cannot be resolved (degenerate axis).",
    "frame_mismatch": "RGB and depth come from different frames.",
    "geometry_exception": "The geometry computation failed.",
    "overlapping_ownership": (
        "Rejected due to overlapping ownership: another detection claims "
        "the same pixels."
    ),
    "ambiguous_surface": "The visible surface is ambiguous.",
    "visible_geometry_mismatch": (
        "The visible 3D geometry does not match the detection."
    ),
    "unsynchronized_input": "RGB and depth inputs were not synchronised.",
    "invalid_hybrid_configuration": (
        "The hybrid RGB-D configuration is invalid."
    ),
    "ambiguous_axis_no_pose": (
        "Orientation cannot be resolved (ambiguous axis), so no pose."
    ),
}

# Non-geometry notes emitted by easy_perception_deployment.hpp.
NOTE_TEXT = {
    "max_detections_cap": (
        "Not published: the Max Detections limit was reached."
    ),
    "not_selected": (
        "Not published: the class is not one of the selected target objects."
    ),
    "no_track_assignment": (
        "Not published: the tracker has not assigned a track yet."
    ),
}

# Short table labels; the Inspector shows the full sentence.
SHORT_TEXT = {
    "invalid_intrinsics": "Bad intrinsics",
    "roi_out_of_bounds": "Box outside image",
    "empty_or_mismatched_mask": "Bad mask",
    "insufficient_depth": "Not enough depth",
    "too_few_segmented_points": "Too few 3D points",
    "nonfinite_geometry": "Non-finite geometry",
    "invalid_dimensions": "Bad dimensions",
    "degenerate_axis": "Orientation unresolved",
    "frame_mismatch": "Frame mismatch",
    "geometry_exception": "Geometry failed",
    "overlapping_ownership": "Overlapping ownership",
    "ambiguous_surface": "Ambiguous surface",
    "visible_geometry_mismatch": "Geometry mismatch",
    "unsynchronized_input": "Unsynchronised input",
    "invalid_hybrid_configuration": "Bad hybrid config",
    "ambiguous_axis_no_pose": "Orientation ambiguous",
    "max_detections_cap": "Max detections cap",
    "not_selected": "Not a target class",
    "no_track_assignment": "No track yet",
}

UNAVAILABLE = "Unavailable"


def short_reason(record):
    """Compact reason for the table: emitted codes, deduplicated."""
    labels = []
    for code in list(record.reasons) + ([record.note] if record.note else []):
        label = SHORT_TEXT.get(code, code)
        if label not in labels:
            labels.append(label)
    return ", ".join(labels)


def reason_text(code):
    """Operator text for an emitted failure name; unknown names stay raw."""
    return REASON_TEXT.get(code, f"Reported reason: {code}")


def note_text(note):
    return NOTE_TEXT.get(note, f"Reported note: {note}")


def stamp_ns_from_parts(sec, nanosec):
    return int(sec) * 1_000_000_000 + int(nanosec)


def format_stamp(stamp_ns):
    """Full-precision source stamp, e.g. ``1792178551.843231445``."""
    if stamp_ns is None:
        return UNAVAILABLE
    sec, nsec = divmod(int(stamp_ns), 1_000_000_000)
    return f"{sec}.{nsec:09d}"


def format_stamp_short(stamp_ns):
    """Compact stamp for tables: last four seconds digits and milliseconds."""
    if stamp_ns is None:
        return "—"
    sec, nsec = divmod(int(stamp_ns), 1_000_000_000)
    return f"…{sec % 10000:04d}.{nsec // 1_000_000:03d} s"


def _attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _level(value):
    if isinstance(value, (bytes, bytearray)):
        return value[0] if value else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _floats(text, count):
    parts = [part.strip() for part in str(text or "").split(",")]
    if len(parts) != count or not all(parts):
        return None
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def _int_or_none(text):
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def _bool_or_none(text):
    value = str(text or "").strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


@dataclass
class DetectionRecord:
    """One detector proposal exactly as the readiness report states it."""

    index: int
    status_name: str
    state: str
    level: int | None
    message: str
    hardware_id: str
    class_name: str
    confidence: float | None
    confidence_text: str
    roi: tuple | None
    mask_pixels: int | None
    valid_depth_pixels: int | None
    position_valid: bool | None
    orientation_valid: bool | None
    centroid_m: tuple | None
    axis: tuple | None
    reasons: tuple
    note: str
    track_id: str | None
    observation_id: str | None
    values: dict
    issues: list = field(default_factory=list)

    @property
    def state_known(self):
        return self.state in STATES

    @property
    def key(self):
        """Identity used to keep a selection across reports."""
        if self.track_id:
            return ("track", self.track_id)
        return ("proposal", self.index)

    def reason_lines(self):
        lines = [reason_text(code) for code in self.reasons]
        if self.note:
            lines.append(note_text(self.note))
        return lines


@dataclass
class ReadinessReport:
    """A parsed DiagnosticArray from the readiness topic."""

    stamp_ns: int
    frame_id: str
    records: list
    observation_id: str | None
    ignored_status_count: int = 0
    issues: list = field(default_factory=list)
    received_monotonic: float | None = None

    def counts(self):
        counts = {state: 0 for state in STATES}
        for record in self.records:
            if record.state in counts:
                counts[record.state] += 1
        return counts


def _parse_status(status):
    name = str(_attr(status, "name", "") or "")
    try:
        index = int(name.rsplit("/", 1)[1])
    except (IndexError, ValueError):
        index = -1
    values = {}
    for item in _attr(status, "values", []) or []:
        values[str(_attr(item, "key", ""))] = str(_attr(item, "value", ""))
    issues = []
    if index < 0:
        issues.append(f"Status name '{name}' has no detection index.")
    state = values.get("state", "").strip()
    if state not in STATES:
        issues.append(f"Unrecognised readiness state '{state}'.")
    message = str(_attr(status, "message", "") or "")
    if state and message and message != state:
        issues.append(
            f"Status message '{message}' differs from state '{state}'.")

    confidence_text = values.get("confidence", "").strip()
    confidence = None
    if confidence_text:
        try:
            confidence = float(confidence_text)
        except ValueError:
            issues.append(f"Confidence '{confidence_text}' is not a number.")
        else:
            if not math.isfinite(confidence):
                confidence = None
                issues.append("Confidence is not finite.")

    roi = None
    roi_text = values.get("roi", "").strip()
    if roi_text:
        parts = roi_text.split(",")
        numbers = [_int_or_none(part) for part in parts]
        if len(numbers) == 4 and None not in numbers:
            roi = tuple(numbers)
        else:
            issues.append(f"ROI '{roi_text}' is malformed.")

    position_valid = _bool_or_none(values.get("position_valid"))
    orientation_valid = _bool_or_none(values.get("orientation_valid"))
    positioned_state = state in ("LOCALISED", "POSE_VALID", "GRASP_CANDIDATE")
    oriented_state = state in ("POSE_VALID", "GRASP_CANDIDATE")

    # Never show geometry beyond what the reported state validates.
    centroid = None
    centroid_text = values.get("centroid_m", "").strip()
    if centroid_text:
        parsed = _floats(centroid_text, 3)
        if not (positioned_state and position_valid):
            issues.append(
                f"Centroid reported for state {state or 'unknown'}; "
                "not shown.")
        elif parsed is None:
            issues.append(f"Centroid '{centroid_text}' is malformed.")
        else:
            centroid = parsed
    elif positioned_state:
        issues.append(f"State {state} reported without a centroid.")

    axis = None
    axis_text = values.get("axis", "").strip()
    if axis_text:
        parsed = _floats(axis_text, 3)
        if not (oriented_state and orientation_valid):
            issues.append(
                f"Axis reported for state {state or 'unknown'}; not shown.")
        elif parsed is None:
            issues.append(f"Axis '{axis_text}' is malformed.")
        else:
            axis = parsed
    elif oriented_state:
        issues.append(f"State {state} reported without an axis.")

    reasons = tuple(
        code.strip()
        for code in values.get("reasons", "").split(",")
        if code.strip() and code.strip() != "none"
    )
    track_id = values.get("track_id", "").strip() or None
    observation_id = values.get("observation_id", "").strip() or None
    return DetectionRecord(
        index=index,
        status_name=name,
        state=state,
        level=_level(_attr(status, "level", None)),
        message=message,
        hardware_id=str(_attr(status, "hardware_id", "") or ""),
        class_name=values.get("class", "").strip(),
        confidence=confidence,
        confidence_text=confidence_text,
        roi=roi,
        mask_pixels=_int_or_none(values.get("mask_pixels")),
        valid_depth_pixels=_int_or_none(values.get("valid_depth_pixels")),
        position_valid=position_valid,
        orientation_valid=orientation_valid,
        centroid_m=centroid,
        axis=axis,
        reasons=reasons,
        note=values.get("note", "").strip(),
        track_id=track_id,
        observation_id=observation_id,
        values=values,
        issues=issues,
    )


def parse_readiness_message(message):
    """Parse a ROS DiagnosticArray, or its JSON dict form, into a report."""
    header = _attr(message, "header")
    stamp = _attr(header, "stamp")
    stamp_ns = stamp_ns_from_parts(
        _attr(stamp, "sec", 0), _attr(stamp, "nanosec", 0))
    frame_id = str(_attr(header, "frame_id", "") or "")
    records = []
    ignored = 0
    for status in _attr(message, "status", []) or []:
        if not str(_attr(status, "name", "")).startswith(STATUS_PREFIX):
            ignored += 1
            continue
        records.append(_parse_status(status))
    records.sort(key=lambda record: record.index)
    observation_ids = {
        record.observation_id for record in records if record.observation_id
    }
    issues = []
    if len(observation_ids) > 1:
        issues.append("Records carry different observation IDs.")
    if stamp_ns == 0:
        issues.append("The report has no source timestamp.")
    return ReadinessReport(
        stamp_ns=stamp_ns,
        frame_id=frame_id,
        records=records,
        observation_id=(
            next(iter(observation_ids)) if len(observation_ids) == 1
            else None),
        ignored_status_count=ignored,
        issues=issues,
    )


def diagnostic_array_to_dict(message):
    """Lossless JSON form of a DiagnosticArray (all strings kept verbatim)."""
    header = message.header
    return {
        "header": {
            "stamp": {
                "sec": int(header.stamp.sec),
                "nanosec": int(header.stamp.nanosec),
            },
            "frame_id": str(header.frame_id),
        },
        "status": [
            {
                "level": _level(status.level),
                "name": str(status.name),
                "message": str(status.message),
                "hardware_id": str(status.hardware_id),
                "values": [
                    {"key": str(item.key), "value": str(item.value)}
                    for item in status.values
                ],
            }
            for status in message.status
        ],
    }


@dataclass
class Association:
    matched: bool
    reason: str


def associate(image_stamp_ns, image_frame_id, report):
    """Pair an RGB frame with a report only on identical source headers.

    The producer stamps each report with the RGB header it was computed
    from, so an exact stamp and frame match is the only trustworthy pairing.
    Being the most recent message is never enough.
    """
    if report is None:
        return Association(False, "No readiness report for this frame.")
    if image_stamp_ns is None:
        return Association(False, "RGB frame missing for this report.")
    if int(image_stamp_ns) != int(report.stamp_ns):
        return Association(
            False,
            "Image and detections do not belong together: report stamp "
            f"{format_stamp(report.stamp_ns)} differs from image stamp "
            f"{format_stamp(image_stamp_ns)}.",
        )
    if str(image_frame_id or "") != report.frame_id:
        return Association(
            False,
            f"Report frame '{report.frame_id}' differs from image frame "
            f"'{image_frame_id}'.",
        )
    return Association(
        True, "Image and report share the same source stamp and frame.")


def roi_inside(roi, width, height):
    if roi is None or not width or not height:
        return False
    x, y, w, h = roi
    return (w > 0 and h > 0 and x >= 0 and y >= 0 and
            x + w <= width and y + h <= height)


# --------------------------------------------------------------------------
# Recorded sessions
# --------------------------------------------------------------------------

SESSION_SCHEMA = "epd.perception_session"
SESSION_SCHEMA_VERSION = 1
SESSION_FILE = "session.json"
PROVENANCE_KINDS = ("recorded", "synthetic")


class SessionError(ValueError):
    """A recording that cannot be replayed truthfully."""


@dataclass
class SessionFrame:
    position: int
    stamp_ns: int
    image_path: Path | None
    image_stamp_ns: int | None
    image_frame_id: str | None
    image_width: int | None
    image_height: int | None
    readiness: ReadinessReport | None
    label: str = ""
    image_sha256: str = ""

    def association(self):
        return associate(self.image_stamp_ns, self.image_frame_id,
                         self.readiness)


@dataclass
class PerceptionSession:
    path: Path
    name: str
    provenance: dict
    frames: list
    readiness_topic: str
    image_topic: str
    warnings: list = field(default_factory=list)

    @property
    def kind(self):
        return self.provenance.get("kind", "")

    @property
    def is_synthetic(self):
        return self.kind == "synthetic"

    @property
    def readiness_frame_count(self):
        return sum(1 for frame in self.frames if frame.readiness is not None)

    def report_rate_hz(self):
        """Report rate measured from consecutive source stamps."""
        stamps = [
            frame.readiness.stamp_ns
            for frame in self.frames if frame.readiness is not None
        ]
        deltas = [
            (b - a) / 1e9 for a, b in zip(stamps, stamps[1:]) if b > a
        ]
        if not deltas:
            return None
        return 1.0 / statistics.median(deltas)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_file(path):
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / SESSION_FILE
    if not path.is_file():
        raise SessionError(f"No {SESSION_FILE} found at {path}.")
    return path


def _relative_file(base, text, what):
    candidate = Path(str(text))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SessionError(f"{what} path must stay inside the recording.")
    resolved = (base / candidate).resolve()
    if not resolved.is_file():
        raise SessionError(f"{what} is missing: {candidate}")
    return resolved


def verify_image_bytes(frame, data):
    """Raise SessionError if a frame's bytes differ from its recorded hash."""
    if frame.image_sha256 and (
            hashlib.sha256(data).hexdigest() != frame.image_sha256):
        raise SessionError(
            f"Frame {frame.position + 1} image does not match its recorded "
            "SHA256; the recording was modified.")


def load_session(path, verify_images=True):
    """Load and verify a recorded perception session manifest.

    With ``verify_images=False`` image hashes are left to the caller, which
    must check each frame with ``verify_image_bytes`` before showing it.
    """
    manifest_path = _session_file(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SessionError(f"Recording manifest is unreadable: {exc}")
    if manifest.get("schema") != SESSION_SCHEMA:
        raise SessionError("Not an EPD perception session manifest.")
    if manifest.get("schema_version") != SESSION_SCHEMA_VERSION:
        raise SessionError(
            "Unsupported session schema_version "
            f"{manifest.get('schema_version')!r}.")
    provenance = dict(manifest.get("provenance") or {})
    if provenance.get("kind") not in PROVENANCE_KINDS:
        raise SessionError(
            "The recording must state whether it is recorded or synthetic.")
    entries = manifest.get("frames") or []
    if not entries:
        raise SessionError("The recording contains no frames.")

    base = manifest_path.parent
    frames = []
    warnings = []
    previous = None
    for position, entry in enumerate(entries):
        image = entry.get("image")
        image_path = image_stamp = image_frame = None
        width = height = None
        if image:
            image_path = _relative_file(base, image.get("path", ""), "Image")
            expected = image.get("sha256") or ""
            if verify_images and expected and _sha256(image_path) != expected:
                raise SessionError(
                    f"Image {image.get('path')} does not match its recorded "
                    "SHA256; the recording was modified.")
            header = image.get("header") or {}
            stamp = header.get("stamp") or {}
            image_stamp = stamp_ns_from_parts(
                stamp.get("sec", 0), stamp.get("nanosec", 0))
            image_frame = str(header.get("frame_id", ""))
            width = _int_or_none(image.get("width"))
            height = _int_or_none(image.get("height"))
        readiness = None
        if entry.get("readiness") is not None:
            readiness = parse_readiness_message(entry["readiness"])
        if image_stamp is None and readiness is None:
            raise SessionError(
                f"Frame {position} has neither an image nor a report.")
        stamp_ns = image_stamp if image_stamp is not None else (
            readiness.stamp_ns)
        if previous is not None and stamp_ns < previous:
            warnings.append(
                f"Frame {position} source stamp goes backwards; it is "
                "replayed without a delay.")
        previous = stamp_ns
        frames.append(SessionFrame(
            position=position,
            stamp_ns=stamp_ns,
            image_path=image_path,
            image_stamp_ns=image_stamp,
            image_frame_id=image_frame,
            image_width=width,
            image_height=height,
            readiness=readiness,
            label=str(entry.get("label", "")),
            image_sha256=str((image or {}).get("sha256") or ""),
        ))
    return PerceptionSession(
        path=manifest_path.parent,
        name=str(manifest.get("name") or manifest_path.parent.name),
        provenance=provenance,
        frames=frames,
        readiness_topic=str(manifest.get("readiness_topic", READINESS_TOPIC)),
        image_topic=str(manifest.get("image_topic", "")),
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Replay timing
# --------------------------------------------------------------------------

class ReplayState:
    EMPTY = "EMPTY"
    READY = "READY"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    ENDED = "ENDED"
    STOPPED = "STOPPED"


class ReplayPlayer:
    """Plays frames at their source-stamp spacing on an injectable clock.

    Gaps longer than ``max_gap_s`` (for example between separately captured
    scenes) are shortened for playback and counted, so the operator is told.
    Source stamps themselves are never altered.
    """

    def __init__(self, max_gap_s=1.5, clock=time.monotonic):
        self._max_gap_s = float(max_gap_s)
        self._clock = clock
        self.unload()

    def unload(self):
        self._offsets = []
        self._hold_s = 0.0
        self.compressed_gaps = 0
        self.state = ReplayState.EMPTY
        self._elapsed = 0.0
        self._started_at = None
        self.index = None

    def load(self, stamps_ns):
        self.unload()
        if not stamps_ns:
            return
        offsets = [0.0]
        deltas = []
        for previous, current in zip(stamps_ns, stamps_ns[1:]):
            delta = max(0.0, (current - previous) / 1e9)
            if delta > self._max_gap_s:
                self.compressed_gaps += 1
                delta = self._max_gap_s
            deltas.append(delta)
            offsets.append(offsets[-1] + delta)
        positive = [delta for delta in deltas if delta > 0]
        self._hold_s = statistics.median(positive) if positive else (
            self._max_gap_s)
        self._offsets = offsets
        self.state = ReplayState.READY

    @property
    def frame_count(self):
        return len(self._offsets)

    @property
    def duration_s(self):
        if not self._offsets:
            return 0.0
        return self._offsets[-1] + self._hold_s

    def elapsed_s(self):
        if self.state == ReplayState.PLAYING and self._started_at is not None:
            return self._elapsed + (self._clock() - self._started_at)
        return self._elapsed

    def play(self):
        if self.state in (ReplayState.EMPTY, ReplayState.PLAYING):
            return
        if self.state in (ReplayState.ENDED, ReplayState.STOPPED):
            self._elapsed = 0.0
        self._started_at = self._clock()
        self.state = ReplayState.PLAYING
        self.tick()

    def pause(self):
        if self.state != ReplayState.PLAYING:
            return
        self._elapsed = self.elapsed_s()
        self._started_at = None
        self.state = ReplayState.PAUSED

    def restart(self):
        if self.state == ReplayState.EMPTY:
            return
        self._elapsed = 0.0
        self._started_at = self._clock()
        self.state = ReplayState.PLAYING
        self.index = None
        self.tick()

    def stop(self):
        if self.state == ReplayState.EMPTY:
            return
        self._elapsed = 0.0
        self._started_at = None
        self.index = None
        self.state = ReplayState.STOPPED

    def step(self, delta):
        """Move one frame while paused or ready."""
        if self.state not in (ReplayState.PAUSED, ReplayState.READY,
                              ReplayState.ENDED):
            return self.index
        current = -1 if self.index is None else self.index
        target = min(max(current + delta, 0), self.frame_count - 1)
        self.index = target
        self._elapsed = self._offsets[target]
        self.state = ReplayState.PAUSED
        return self.index

    def tick(self):
        """Advance the timeline; return the frame index to display."""
        if self.state != ReplayState.PLAYING:
            return self.index
        elapsed = self.elapsed_s()
        index = 0
        for position, offset in enumerate(self._offsets):
            if offset <= elapsed:
                index = position
            else:
                break
        self.index = index
        if elapsed >= self.duration_s:
            self._elapsed = self.duration_s
            self._started_at = None
            self.state = ReplayState.ENDED
        return self.index

    def frame_elapsed_s(self):
        """Replay-clock time since the displayed frame was reached."""
        if self.index is None:
            return None
        return max(0.0, self.elapsed_s() - self._offsets[self.index])


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------

HEALTH_STATES = (
    "STOPPED",
    "STARTING",
    "LIVE",
    "REPLAY ACTIVE",
    "WAITING",
    "STALE",
    "FAILED",
    "UNAVAILABLE",
)


@dataclass
class Health:
    state: str
    headline: str
    messages: list = field(default_factory=list)


def replay_health(player_state, session, frame, load_error=""):
    if load_error:
        return Health("FAILED", "The recording could not be loaded.",
                      [load_error])
    if session is None or player_state == ReplayState.EMPTY:
        return Health("STOPPED", "No recording loaded.")
    messages = []
    if session.readiness_frame_count == 0:
        messages.append(
            "Readiness topic unavailable in this recording. EPD publishes "
            "it only with hybrid RGB-D refinement enabled.")
    if frame is not None:
        if frame.readiness is None:
            messages.append("No readiness report for this frame.")
        if frame.image_path is None:
            messages.append("RGB frame missing for this report.")
        elif frame.readiness is not None:
            association = frame.association()
            if not association.matched:
                messages.append(association.reason)
    if player_state == ReplayState.STOPPED:
        return Health("STOPPED", "Replay stopped. No detections shown.",
                      messages)
    if player_state == ReplayState.READY:
        return Health("STOPPED", "Recording loaded. Press Play.", messages)
    if player_state == ReplayState.ENDED:
        return Health(
            "STALE",
            "Replay finished. The last recorded frame is no longer "
            "updating.",
            messages)
    if player_state == ReplayState.PAUSED:
        return Health("REPLAY ACTIVE", "Replay paused.", messages)
    return Health("REPLAY ACTIVE", "Replaying a recording.", messages)


def stale_after_s(arrival_times):
    """Staleness limit: three typical report periods, at least 5 s."""
    deltas = [
        b - a for a, b in zip(arrival_times, arrival_times[1:]) if b > a
    ]
    if not deltas:
        return 5.0
    return max(5.0, 3.0 * statistics.median(deltas))


def live_health(job_state, ros_error, publisher_count, last_report_age,
                stale_limit_s, running_for_s):
    """Health of the live readiness topic.

    ``job_state`` is the Deploy job state name. ``last_report_age`` is the
    monotonic time since the last report arrived, or None.
    """
    if ros_error:
        return Health("UNAVAILABLE", "ROS readiness monitor unavailable.",
                      [ros_error])
    fresh = (last_report_age is not None and
             last_report_age <= stale_limit_s)
    if job_state == "FAILED":
        return Health("FAILED", "Deployment failed.",
                      ["Review the runtime message and deploy log."])
    if job_state == "STARTING":
        return Health("STARTING", "Perception is starting.")
    if job_state == "STOPPING":
        return Health("STOPPED", "Perception is stopping.")
    if job_state != "RUNNING":
        if fresh:
            return Health(
                "LIVE", "Readiness reports are arriving.",
                ["The publishing EPD node was not started from this "
                 "window."])
        if last_report_age is not None:
            return Health(
                "STALE", "Observation too old.",
                [f"The last readiness report arrived {last_report_age:.1f} "
                 "s ago and nothing is publishing now."])
        return Health("STOPPED", "Perception is stopped.")
    if last_report_age is None:
        if publisher_count == 0 and running_for_s > 5.0:
            return Health(
                "UNAVAILABLE", "Readiness topic unavailable.",
                ["EPD publishes per-detection readiness only with hybrid "
                 "RGB-D refinement enabled (Localization or Tracking)."])
        return Health("WAITING", "No perception output received yet.")
    if not fresh:
        return Health(
            "STALE", "Observation too old.",
            [f"The last readiness report arrived {last_report_age:.1f} s "
             "ago."])
    return Health("LIVE", "Readiness reports are arriving.")
