import importlib.util
import json
import stat
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def _load(name):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DIAGNOSTICS = _load("epd_diagnostics_bundle")
ACCEPTANCE = _load("epd_release_acceptance")


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _package(tmp_path, with_assets=True):
    root = tmp_path / "package"
    _write_json(
        root / "config" / "session_config.json",
        {
            "schema_version": 2,
            "path_to_model": "./data/model/model.onnx",
            "path_to_label_list": "./data/label_list/labels.txt",
            "execution_backend": "cpu",
        },
    )
    _write_json(
        root / "config" / "usecase_config.json",
        {"schema_version": 2, "usecase_mode": 4, "track_type": "MEDIANFLOW"},
    )
    _write_json(
        root / "config" / "input_image_topic.json",
        {
            "schema_version": 2,
            "input_image_topic": "/camera/camera/color/image_raw",
        },
    )
    for relative in (
        "launch/run.launch.py",
        "launch/replay.launch.py",
        "launch/workcell_contract.launch.py",
        "scripts/epd_diagnostics_bundle.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    if with_assets:
        model = root / "data" / "model" / "model.onnx"
        labels = root / "data" / "label_list" / "labels.txt"
        model.parent.mkdir(parents=True, exist_ok=True)
        labels.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"model")
        labels.write_text("part\n", encoding="utf-8")
    return root


def test_release_static_checks_pass_with_complete_fixture(tmp_path, monkeypatch):
    root = _package(tmp_path)
    monkeypatch.setenv("ROS_DISTRO", "humble")
    results = ACCEPTANCE.static_checks(root)
    assert ACCEPTANCE.overall_status(results) == "PASS"
    assert all(result["status"] == "PASS" for result in results)


def test_release_static_checks_fail_for_missing_model(tmp_path, monkeypatch):
    root = _package(tmp_path, with_assets=False)
    monkeypatch.setenv("ROS_DISTRO", "humble")
    results = ACCEPTANCE.static_checks(root)
    assert ACCEPTANCE.overall_status(results) == "FAIL"
    model = next(item for item in results if item["name"] == "ONNX model asset")
    assert model["status"] == "FAIL"


def test_warn_does_not_turn_overall_result_into_fail():
    results = [
        ACCEPTANCE.check("required", "PASS", "ok"),
        ACCEPTANCE.check("optional", "WARN", "not connected"),
    ]
    assert ACCEPTANCE.overall_status(results) == "WARN"


def test_redactor_hides_home_and_user(monkeypatch):
    monkeypatch.setenv("USER", "epd-user")
    redact = DIAGNOSTICS.redactor(include_paths=False)
    text = redact(str(Path.home()) + "/ws by epd-user")
    assert str(Path.home()) not in text
    assert "epd-user" not in text
    assert "<HOME>" in text
    assert "<USER>" in text


def test_diagnostics_collect_records_unavailable_commands(tmp_path, monkeypatch):
    root = _package(tmp_path)

    def unavailable(command, timeout=8.0, env=None):
        return {
            "command": command,
            "returncode": None,
            "duration_s": 0.0,
            "error": "tool unavailable",
        }

    monkeypatch.setattr(DIAGNOSTICS, "command_result", unavailable)
    bundle = tmp_path / "bundle"
    manifest = DIAGNOSTICS.collect(root, bundle)
    assert manifest["schema_version"] == DIAGNOSTICS.SCHEMA
    assert manifest["read_only_collection"] is True
    assert (bundle / "manifest.json").is_file()
    assert manifest["commands"]["ros2_topics"]["error"] == "tool unavailable"


def test_reference_profile_is_importable_shape():
    outer_package = Path(__file__).resolve().parents[2]
    profile = (
        outer_package
        / "examples"
        / "profiles"
        / "realsense_tracking_cpu.epd-profile.json"
    )
    payload = json.loads(profile.read_text(encoding="utf-8"))
    assert payload["profile_schema_version"] == 1
    assert payload["epd"]["usecase_config"]["usecase_mode"] == 4
    assert payload["epd"]["session_config"]["execution_backend"] == "cpu"


def test_release_helpers_are_executable_for_ros2_run():
    for name in ("epd_release_acceptance.py", "epd_diagnostics_bundle.py"):
        mode = (SCRIPTS / name).stat().st_mode
        assert mode & stat.S_IXUSR
