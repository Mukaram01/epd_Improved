import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "launch"
    / "workcell_contract_bridge.py"
)
SPEC = importlib.util.spec_from_file_location("workcell_contract_bridge", SCRIPT)
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


def _stamp(sec=10, nanosec=25):
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def _point(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def _object(name="part", quaternion=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(
        name=name,
        centroid=_point(0.1, -0.2, 0.7),
        length=0.08,
        breadth=0.04,
        height=0.03,
        axis=_point(1.0, 0.0, 0.0),
        pose=SimpleNamespace(
            position=_point(0.1, -0.2, 0.7),
            orientation=SimpleNamespace(
                x=quaternion[0],
                y=quaternion[1],
                z=quaternion[2],
                w=quaternion[3],
            ),
        ),
        roi=SimpleNamespace(x_offset=10, y_offset=20, width=30, height=40),
        segmented_pcl=SimpleNamespace(width=15, height=1),
    )


def _tracking_message(objects=None, ids=None):
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id="camera_color_optical_frame",
            stamp=_stamp(),
        ),
        objects=objects if objects is not None else [_object()],
        object_ids=ids if ids is not None else ["42"],
        lost_track_ids=["7"],
        process_time=37,
    )


def _localization_message(obj=None):
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id="camera_color_optical_frame",
            stamp=_stamp(),
        ),
        objects=[obj or _object()],
        process_time=31,
    )


def test_tracking_snapshot_preserves_stable_identity_geometry_and_profile():
    snapshot, errors, warnings = BRIDGE.build_snapshot(
        _tracking_message(),
        source_kind="tracking",
        scene_id="ur5_2f_test",
        camera_id="realsense_d435i_1",
        profile_ref="D435i Table Pick",
        runtime_mode="live",
        require_tracking_ids=True,
    )

    assert errors == []
    assert warnings == []
    assert snapshot["schema_version"] == "workcell_perception_snapshot/v1"
    assert snapshot["scene_id"] == "ur5_2f_test"
    assert snapshot["camera_id"] == "realsense_d435i_1"
    assert snapshot["profile_ref"] == "D435i Table Pick"
    assert snapshot["runtime_mode"] == "live"
    assert snapshot["timestamp_ns"] == 10_000_000_025
    assert snapshot["frame_id"] == "camera_color_optical_frame"
    assert snapshot["lost_object_ids"] == ["7"]

    observed = snapshot["objects"][0]
    assert observed["object_id"] == "42"
    assert observed["track_id"] == "42"
    assert observed["label"] == "part"
    assert observed["dimensions_xyz"] == [0.08, 0.04, 0.03]
    assert observed["shape"] == "box"
    assert observed["pose"]["position"] == [0.1, -0.2, 0.7]
    assert observed["pose"]["orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert observed["attributes"]["identity_scope"] == "tracking"
    assert observed["attributes"]["segmented_cloud_points"] == 15
    assert "confidence" not in observed
    assert observed["attributes"]["confidence_available"] is False
    assert BRIDGE.validate_snapshot(snapshot) == []


def test_localization_uses_observation_identity_and_centroid_fallback():
    obj = _object(quaternion=(0.0, 0.0, 0.0, 0.0))
    snapshot, errors, warnings = BRIDGE.build_snapshot(
        _localization_message(obj),
        source_kind="localization",
        scene_id="scene_a",
        camera_id="cam_a",
        runtime_mode="replay",
        require_tracking_ids=False,
    )

    assert errors == []
    assert warnings
    observed = snapshot["objects"][0]
    assert observed["object_id"].startswith("localization:10000000025:0")
    assert "track_id" not in observed
    assert "pose" not in observed
    assert observed["centroid"] == [0.1, -0.2, 0.7]
    assert observed["attributes"]["identity_scope"] == "observation"
    assert snapshot["runtime_mode"] == "replay"


def test_require_tracking_ids_blocks_localization_only_objects():
    _snapshot, errors, _warnings = BRIDGE.build_snapshot(
        _localization_message(),
        source_kind="localization",
        scene_id="scene_a",
        camera_id="cam_a",
        require_tracking_ids=True,
    )
    assert any("stable Tracking ID" in error for error in errors)


def test_snapshot_validator_rejects_duplicate_identity():
    snapshot, errors, _warnings = BRIDGE.build_snapshot(
        _tracking_message(objects=[_object(), _object()], ids=["9", "9"]),
        source_kind="tracking",
        scene_id="scene_a",
        camera_id="cam_a",
    )
    assert any("duplicate object id" in error for error in errors)
    assert any("duplicate object id" in error for error in BRIDGE.validate_snapshot(snapshot))


def test_status_truth_transitions_waiting_ready_stale_and_failed():
    base = dict(
        scene_id="scene_a",
        camera_id="cam_a",
        profile_ref="profile_a",
        runtime_mode="live",
        source_mode="tracking",
        stale_timeout_s=2.0,
        backend={"available": False, "level": None, "message": "", "values": {}},
    )
    waiting = BRIDGE.status_payload(
        **base,
        last_snapshot=None,
        last_message_age_s=None,
    )
    assert waiting["state"] == "WAITING"

    snapshot, errors, _warnings = BRIDGE.build_snapshot(
        _tracking_message(),
        source_kind="tracking",
        scene_id="scene_a",
        camera_id="cam_a",
    )
    assert errors == []
    ready = BRIDGE.status_payload(
        **base,
        last_snapshot=snapshot,
        last_message_age_s=0.1,
    )
    assert ready["state"] == "READY"
    assert ready["stable_tracking_ids"] is True
    assert ready["object_count"] == 1

    stale = BRIDGE.status_payload(
        **base,
        last_snapshot=snapshot,
        last_message_age_s=2.1,
    )
    assert stale["state"] == "STALE"

    failed = BRIDGE.status_payload(
        **base,
        last_snapshot=snapshot,
        last_message_age_s=0.1,
        last_error="source timestamp moved backward",
    )
    assert failed["state"] == "FAILED"


def test_backend_error_is_reported_without_rewriting_snapshot():
    snapshot, errors, _warnings = BRIDGE.build_snapshot(
        _tracking_message(),
        source_kind="tracking",
        scene_id="scene_a",
        camera_id="cam_a",
    )
    assert errors == []
    status = BRIDGE.status_payload(
        scene_id="scene_a",
        camera_id="cam_a",
        profile_ref="",
        runtime_mode="live",
        source_mode="tracking",
        last_snapshot=snapshot,
        last_message_age_s=0.1,
        stale_timeout_s=2.0,
        backend={
            "available": True,
            "level": 2,
            "message": "inference worker error",
            "values": {"inference_failed": "1"},
        },
    )
    assert status["state"] == "FAILED"
    assert status["backend"]["values"]["inference_failed"] == "1"
