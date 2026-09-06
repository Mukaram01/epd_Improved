"""Align Smart Model Manager output validation with EPD P2/P3 runtime semantics.

EPD's detector post-processing has a fixed output contract:

P2: boxes(float32), labels(int64), scores(float32)
P3: boxes(float32), labels(int64), scores(float32), masks(float32)

The original EPD-3 preflight treated every output as float32, which incorrectly
blocked the bundled Faster R-CNN / Mask R-CNN ONNX models because their class
label tensor is correctly INT64 (ONNX element type 7).
"""

from __future__ import annotations

from windows import model_manager


ONNX_FLOAT = 1
ONNX_INT64 = 7

_OUTPUT_CONTRACTS = {
    3: (
        ("boxes", ONNX_FLOAT),
        ("labels", ONNX_INT64),
        ("scores", ONNX_FLOAT),
    ),
    4: (
        ("boxes", ONNX_FLOAT),
        ("labels", ONNX_INT64),
        ("scores", ONNX_FLOAT),
        ("masks", ONNX_FLOAT),
    ),
}


def validate_epd_outputs(metadata, blockers):
    """Validate the exact tensor types consumed by EPD P2/P3 post-processing."""
    output_count = int(metadata.get("output_count", 0) or 0)
    profile = model_manager.precision_profile(output_count)

    contract = _OUTPUT_CONTRACTS.get(output_count)
    if contract is None:
        if output_count == 1:
            blockers.append(
                "Precision Level 1 deployment is deprecated in the current EPD runtime. "
                "Use a P2 (3-output) or P3 (4-output) model."
            )
        else:
            blockers.append(
                "EPD identifies models by output count: P2 requires 3 outputs and "
                f"P3 requires 4; this model has {output_count}."
            )
        return profile

    outputs = list(metadata.get("outputs", []) or [])
    if len(outputs) != output_count:
        blockers.append(
            "Inspector output metadata is incomplete: "
            f"declared {output_count} outputs but described {len(outputs)}."
        )
        return profile

    for index, ((semantic_name, expected_type), output) in enumerate(
        zip(contract, outputs)
    ):
        if not output.get("tensor", False):
            blockers.append(f"Model output {index + 1} ({semantic_name}) is not a tensor.")
            continue

        actual_type = output.get("element_type")
        if actual_type == expected_type:
            continue

        expected_name = "float32" if expected_type == ONNX_FLOAT else "int64"
        blockers.append(
            f"Model output {index + 1} ({semantic_name}) must be ONNX {expected_name} "
            f"(element type {expected_type}) for EPD P{profile['precision_level']} "
            f"post-processing; got element type {actual_type}."
        )

    return profile


def apply_model_output_contract():
    """Install the corrected validator once, before Deploy model preflight runs."""
    current = model_manager._validate_outputs
    if getattr(current, "_epd_output_contract_truth", False):
        return current

    validate_epd_outputs._epd_output_contract_truth = True
    validate_epd_outputs._epd_original_validator = current
    model_manager._validate_outputs = validate_epd_outputs
    return validate_epd_outputs
