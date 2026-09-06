import hashlib
import json
from pathlib import Path

import windows.model_manager as model_manager


def _write_model(tmp_path, content=b"fake onnx bytes for inspection"):
    model_path = tmp_path / "custom.onnx"
    model_path.write_bytes(content)
    return model_path


def _write_labels(tmp_path, labels=None):
    labels = labels or ["background", "part", "tool"]
    label_path = tmp_path / "labels.txt"
    label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return label_path


def _metadata(output_count=3, input_count=1, input_shape=None):
    input_shape = input_shape or [3, -1, -1]
    return {
        "valid": True,
        "input_count": input_count,
        "output_count": output_count,
        "inputs": [
            {
                "name": "image",
                "tensor": True,
                "element_type": 1,
                "rank": len(input_shape),
                "shape": input_shape,
            }
        ],
        "outputs": [
            {
                "name": f"output_{index}",
                "tensor": True,
                "element_type": 1,
                "rank": 1,
                "shape": [-1],
            }
            for index in range(output_count)
        ],
    }


def test_precision_profile_matches_epd_output_contract():
    assert model_manager.precision_profile(1)["precision_level"] == 1
    assert model_manager.precision_profile(3)["precision_level"] == 2
    assert model_manager.precision_profile(4)["precision_level"] == 3
    assert model_manager.precision_profile(2)["deploy_supported"] is False


def test_custom_p2_model_is_ready_for_counting(monkeypatch, tmp_path):
    model_path = _write_model(tmp_path)
    label_path = _write_labels(tmp_path)
    monkeypatch.setattr(
        model_manager,
        "run_runtime_inspector",
        lambda path: _metadata(output_count=3),
    )

    result = model_manager.inspect_deployment_model(
        model_path,
        label_path,
        usecase_mode=1,
        package_root=tmp_path,
    )

    assert result["status"] == "ready"
    assert result["precision_level"] == 2
    assert "Counting" in result["supported_modes"]
    assert any("order cannot be proven" in item for item in result["warnings"])


def test_p2_model_blocks_localization(monkeypatch, tmp_path):
    model_path = _write_model(tmp_path)
    label_path = _write_labels(tmp_path)
    monkeypatch.setattr(
        model_manager,
        "run_runtime_inspector",
        lambda path: _metadata(output_count=3),
    )

    result = model_manager.inspect_deployment_model(
        model_path,
        label_path,
        usecase_mode=3,
        package_root=tmp_path,
    )

    assert result["status"] == "blocked"
    assert any("Precision Level 3" in item for item in result["blockers"])


def test_p3_model_is_ready_for_tracking(monkeypatch, tmp_path):
    model_path = _write_model(tmp_path)
    label_path = _write_labels(tmp_path)
    monkeypatch.setattr(
        model_manager,
        "run_runtime_inspector",
        lambda path: _metadata(output_count=4),
    )

    result = model_manager.inspect_deployment_model(
        model_path,
        label_path,
        usecase_mode=4,
        package_root=tmp_path,
    )

    assert result["status"] == "ready"
    assert result["precision_level"] == 3
    assert "Tracking" in result["supported_modes"]


def test_legacy_p1_is_explicitly_blocked(monkeypatch, tmp_path):
    model_path = _write_model(tmp_path)
    label_path = _write_labels(tmp_path)
    monkeypatch.setattr(
        model_manager,
        "run_runtime_inspector",
        lambda path: _metadata(output_count=1),
    )

    result = model_manager.inspect_deployment_model(
        model_path,
        label_path,
        usecase_mode=0,
        package_root=tmp_path,
    )

    assert result["status"] == "blocked"
    assert any("deprecated" in item for item in result["blockers"])


def test_multiple_inputs_are_blocked(monkeypatch, tmp_path):
    model_path = _write_model(tmp_path)
    label_path = _write_labels(tmp_path)
    metadata = _metadata(output_count=3, input_count=2)
    monkeypatch.setattr(model_manager, "run_runtime_inspector", lambda path: metadata)

    result = model_manager.inspect_deployment_model(
        model_path,
        label_path,
        usecase_mode=1,
        package_root=tmp_path,
    )

    assert result["status"] == "blocked"
    assert any("exactly one model input" in item for item in result["blockers"])


def test_non_float_output_is_blocked(monkeypatch, tmp_path):
    model_path = _write_model(tmp_path)
    label_path = _write_labels(tmp_path)
    metadata = _metadata(output_count=3)
    metadata["outputs"][1]["element_type"] = 7
    monkeypatch.setattr(model_manager, "run_runtime_inspector", lambda path: metadata)

    result = model_manager.inspect_deployment_model(
        model_path,
        label_path,
        usecase_mode=1,
        package_root=tmp_path,
    )

    assert result["status"] == "blocked"
    assert any("reads outputs as float32" in item for item in result["blockers"])


def test_trusted_catalog_fallback_verifies_exact_label_order(monkeypatch, tmp_path):
    package_root = tmp_path / "package"
    model_dir = package_root / "data" / "model"
    label_dir = package_root / "data" / "label_list"
    model_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    model_path = model_dir / "trusted.onnx"
    model_path.write_bytes(b"trusted model payload")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    canonical = ["background", "part", "tool"]
    canonical_path = label_dir / "canonical.txt"
    canonical_path.write_text("\n".join(canonical) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "models": [
            {
                "id": "trusted",
                "name": "Trusted P3",
                "filename": model_path.name,
                "sha256": digest,
                "task": "Instance segmentation",
                "precision_level": 3,
                "output_count": 4,
                "input_rank": 3,
                "input_layout": "CHW",
                "canonical_labels": "canonical.txt",
                "label_count": 3,
                "recommended_modes": ["Tracking"],
            }
        ],
    }
    (model_dir / "model_library.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    selected_labels = tmp_path / "selected.txt"
    selected_labels.write_text("\n".join(canonical) + "\n", encoding="utf-8")

    def unavailable(path):
        raise model_manager.ModelInspectionError("helper missing")

    monkeypatch.setattr(model_manager, "run_runtime_inspector", unavailable)
    result = model_manager.inspect_deployment_model(
        model_path,
        selected_labels,
        usecase_mode=4,
        package_root=package_root,
    )

    assert result["status"] == "ready"
    assert result["inspection_source"] == "trusted SHA256 catalog"
    assert result["labels"]["state"] == "verified"


def test_trusted_model_blocks_wrong_label_order(monkeypatch, tmp_path):
    package_root = tmp_path / "package"
    model_dir = package_root / "data" / "model"
    label_dir = package_root / "data" / "label_list"
    model_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    model_path = model_dir / "trusted.onnx"
    model_path.write_bytes(b"trusted model payload")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (label_dir / "canonical.txt").write_text(
        "background\npart\ntool\n",
        encoding="utf-8",
    )
    (model_dir / "model_library.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "Trusted P2",
                        "filename": model_path.name,
                        "sha256": digest,
                        "task": "Object detection",
                        "precision_level": 2,
                        "output_count": 3,
                        "input_rank": 3,
                        "input_layout": "CHW",
                        "canonical_labels": "canonical.txt",
                        "label_count": 3,
                        "recommended_modes": ["Counting"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    wrong_labels = tmp_path / "wrong.txt"
    wrong_labels.write_text("background\ntool\npart\n", encoding="utf-8")

    monkeypatch.setattr(
        model_manager,
        "run_runtime_inspector",
        lambda path: (_ for _ in ()).throw(
            model_manager.ModelInspectionError("helper missing")
        ),
    )
    result = model_manager.inspect_deployment_model(
        model_path,
        wrong_labels,
        usecase_mode=1,
        package_root=package_root,
    )

    assert result["status"] == "blocked"
    assert any("Label order/content" in item for item in result["blockers"])


def test_unknown_custom_model_requires_runtime_inspector(monkeypatch, tmp_path):
    model_path = _write_model(tmp_path)
    label_path = _write_labels(tmp_path)

    def unavailable(path):
        raise model_manager.ModelInspectionError("not installed")

    monkeypatch.setattr(model_manager, "run_runtime_inspector", unavailable)
    result = model_manager.inspect_deployment_model(
        model_path,
        label_path,
        usecase_mode=1,
        package_root=tmp_path,
    )

    assert result["status"] == "blocked"
    assert any("could not be inspected" in item for item in result["blockers"])
