from windows import model_manager
from windows.model_output_contract import (
    ONNX_FLOAT,
    ONNX_INT64,
    apply_model_output_contract,
    validate_epd_outputs,
)


def _metadata(types):
    return {
        "output_count": len(types),
        "outputs": [
            {
                "name": f"output_{index}",
                "tensor": True,
                "element_type": element_type,
            }
            for index, element_type in enumerate(types)
        ],
    }


def test_p3_maskrcnn_contract_accepts_int64_labels():
    blockers = []
    profile = validate_epd_outputs(
        _metadata([ONNX_FLOAT, ONNX_INT64, ONNX_FLOAT, ONNX_FLOAT]),
        blockers,
    )

    assert blockers == []
    assert profile["precision_level"] == 3
    assert profile["deploy_supported"] is True


def test_p2_fasterrcnn_contract_accepts_int64_labels():
    blockers = []
    profile = validate_epd_outputs(
        _metadata([ONNX_FLOAT, ONNX_INT64, ONNX_FLOAT]),
        blockers,
    )

    assert blockers == []
    assert profile["precision_level"] == 2
    assert profile["deploy_supported"] is True


def test_float32_label_tensor_is_blocked_because_runtime_reads_int64_labels():
    blockers = []
    validate_epd_outputs(
        _metadata([ONNX_FLOAT, ONNX_FLOAT, ONNX_FLOAT, ONNX_FLOAT]),
        blockers,
    )

    assert len(blockers) == 1
    assert "output 2 (labels)" in blockers[0].lower()
    assert "int64" in blockers[0].lower()


def test_non_float_mask_tensor_is_blocked():
    blockers = []
    validate_epd_outputs(
        _metadata([ONNX_FLOAT, ONNX_INT64, ONNX_FLOAT, ONNX_INT64]),
        blockers,
    )

    assert len(blockers) == 1
    assert "output 4 (masks)" in blockers[0].lower()
    assert "float32" in blockers[0].lower()


def test_incomplete_output_metadata_is_blocked():
    blockers = []
    metadata = _metadata([ONNX_FLOAT, ONNX_INT64, ONNX_FLOAT, ONNX_FLOAT])
    metadata["outputs"].pop()

    validate_epd_outputs(metadata, blockers)

    assert len(blockers) == 1
    assert "metadata is incomplete" in blockers[0].lower()


def test_patch_installation_is_idempotent(monkeypatch):
    original = model_manager._validate_outputs
    monkeypatch.setattr(model_manager, "_validate_outputs", original)

    first = apply_model_output_contract()
    second = apply_model_output_contract()

    assert first is validate_epd_outputs
    assert second is validate_epd_outputs
    assert model_manager._validate_outputs is validate_epd_outputs
