from types import SimpleNamespace

from windows.three_d_diagnostics import (
    diagnostics_values,
    inspect_frame_contract,
    inspect_object,
    sample_depth_validity,
    summarize_p3_message,
)


def _stamp(sec=1, nanosec=2):
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def _point(x=0.1, y=0.2, z=0.8):
    return SimpleNamespace(x=x, y=y, z=z)


def _pose():
    return SimpleNamespace(
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    )


def _depth(values=(1000, 0, 1200, 1300), stamp=None):
    raw = b"".join(int(value).to_bytes(2, "little") for value in values)
    return SimpleNamespace(
        width=2,
        height=2,
        encoding="16UC1",
        is_bigendian=0,
        data=raw,
        header=SimpleNamespace(stamp=stamp or _stamp()),
    )


def _object():
    return SimpleNamespace(
        name="part",
        centroid=_point(),
        length=0.10,
        breadth=0.05,
        height=0.03,
        segmented_pcl=SimpleNamespace(width=25, height=1),
        axis=_point(1.0, 0.0, 0.0),
        pose=_pose(),
    )


def _message(tracking=False):
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(), frame_id="camera_color_optical_frame"),
        frame_width=2,
        frame_height=2,
        depth_image=_depth(),
        fx=610.4,
        fy=610.2,
        ppx=323.3,
        ppy=237.8,
        process_time=42,
        objects=[_object()],
    )
    if tracking:
        message.object_ids = ["7"]
        message.lost_track_ids = ["4"]
    return message


def test_frame_contract_reports_exact_alignment():
    result = inspect_frame_contract(_message())
    assert result["state"] == "aligned"
    assert result["intrinsics_ok"] is True
    assert result["shape_ok"] is True
    assert result["stamps_match"] is True


def test_frame_contract_rejects_bad_intrinsics_and_shape():
    message = _message()
    message.fx = 0.0
    message.depth_image.width = 3
    result = inspect_frame_contract(message)
    assert result["state"] == "invalid"
    assert result["intrinsics_ok"] is False
    assert result["shape_ok"] is False


def test_sample_depth_validity_is_explicitly_sampled():
    result = sample_depth_validity(_depth(), max_samples=100)
    assert result["supported"] is True
    assert result["samples"] == 4
    assert result["valid"] == 3
    assert result["ratio"] == 0.75


def test_object_inspector_uses_geometry_truth():
    result = inspect_object(_object(), "7")
    assert result["id"] == "7"
    assert result["name"] == "part"
    assert result["cloud_points"] == 25
    assert result["inspector_state"] == "valid"


def test_object_without_valid_centroid_is_invalid():
    obj = _object()
    obj.centroid.z = float("nan")
    assert inspect_object(obj)["inspector_state"] == "invalid"


def test_tracking_summary_keeps_current_and_lost_ids():
    result = summarize_p3_message(_message(tracking=True), tracking=True)
    assert result["tracking"] is True
    assert result["object_ids"] == ["7"]
    assert result["lost_track_ids"] == ["4"]
    assert result["objects"][0]["id"] == "7"


def test_diagnostics_values_selects_inference_worker():
    message = SimpleNamespace(
        status=[
            SimpleNamespace(name="other", values=[]),
            SimpleNamespace(
                name="easy_perception_deployment/inference_worker",
                values=[
                    SimpleNamespace(key="geometry_valid_total", value="12"),
                    SimpleNamespace(key="insufficient_depth_total", value="2"),
                ],
            ),
        ]
    )
    result = diagnostics_values(message)
    assert result["geometry_valid_total"] == "12"
    assert result["insufficient_depth_total"] == "2"
