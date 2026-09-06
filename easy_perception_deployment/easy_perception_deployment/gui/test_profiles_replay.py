import json
from pathlib import Path

from windows.epd5_productization import (
    parse_rosbag_topics,
    replay_command,
    rosbag_play_command,
)
from windows.profile_store import (
    ProfileStore,
    apply_profile_to_files,
    capture_profile,
    profile_summary,
)


def _package_root(tmp_path):
    root = tmp_path / "package"
    (root / "config").mkdir(parents=True)
    (root / "data" / "model").mkdir(parents=True)
    (root / "data" / "label_list").mkdir(parents=True)
    model = root / "data" / "model" / "parts.onnx"
    labels = root / "data" / "label_list" / "parts.txt"
    model.write_bytes(b"fake-onnx-for-profile-test")
    labels.write_text("background\npart\n", encoding="utf-8")
    (root / "config" / "session_config.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "path_to_model": "./data/model/parts.onnx",
                "path_to_label_list": "./data/label_list/parts.txt",
                "visualizeFlag": "visualize",
                "useCPU": "CPU",
                "intra_op_num_threads": 2,
                "image_transport": "raw",
                "publish_detection_segmentation": True,
                "confidence_threshold": 0.65,
                "max_detections": 50,
            }
        ),
        encoding="utf-8",
    )
    (root / "config" / "usecase_config.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "usecase_mode": 4,
                "track_type": "MEDIANFLOW",
            }
        ),
        encoding="utf-8",
    )
    (root / "config" / "input_image_topic.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "input_image_topic": "/camera/camera/color/image_raw",
            }
        ),
        encoding="utf-8",
    )
    return root


def test_capture_profile_contains_reproducible_runtime_truth(tmp_path):
    root = _package_root(tmp_path)
    profile = capture_profile(root, "Table Pick", "Known-good D435i setup")
    summary = profile_summary(profile)
    assert profile["profile_schema_version"] == 1
    assert profile["assets"]["model"]["sha256"]
    assert profile["assets"]["labels"]["sha256"]
    assert summary["mode"] == "Tracking"
    assert summary["topic"] == "/camera/camera/color/image_raw"
    assert summary["confidence"] == 0.65


def test_profile_store_known_good_round_trip(tmp_path):
    root = _package_root(tmp_path)
    store = ProfileStore(tmp_path / "profiles")
    path = store.save(capture_profile(root, "Known Good"))
    store.set_known_good(path)
    known_path, profile = store.restore_known_good()
    assert known_path == path
    assert profile["name"] == "Known Good"


def test_apply_profile_relocates_asset_by_verified_hash(tmp_path):
    source = _package_root(tmp_path / "source")
    profile = capture_profile(source, "Portable")
    profile["epd"]["session_config"]["path_to_model"] = "/missing/parts.onnx"
    profile["epd"]["session_config"]["path_to_label_list"] = "/missing/parts.txt"

    target = _package_root(tmp_path / "target")
    applied, warnings = apply_profile_to_files(profile, target)
    session = applied["epd"]["session_config"]
    assert session["path_to_model"] == "./data/model/parts.onnx"
    assert session["path_to_label_list"] == "./data/label_list/parts.txt"
    assert len(warnings) == 2


def test_parse_rosbag_topics_from_standard_info_output():
    output = """
    Topic information: Topic: /camera/color/image_raw | Type: sensor_msgs/msg/Image | Count: 10
    Topic information: Topic: /camera/color/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 10
    """
    assert parse_rosbag_topics(output) == [
        "/camera/color/camera_info",
        "/camera/color/image_raw",
    ]


def test_replay_commands_are_argument_lists(tmp_path):
    fixture = tmp_path / "fixture.json"
    summary = tmp_path / "summary.json"
    bag = tmp_path / "bag"
    command = replay_command(fixture, "fast", summary)
    assert command[:4] == [
        "ros2",
        "launch",
        "easy_perception_deployment",
        "replay.launch.py",
    ]
    assert f"fixture:={fixture}" in command
    assert rosbag_play_command(bag) == ["ros2", "bag", "play", str(bag)]
