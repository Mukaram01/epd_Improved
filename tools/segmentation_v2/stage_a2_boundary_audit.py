#!/usr/bin/env python3
"""Read-only Stage-A2 28/56-grid mask boundary audit from retained RGB-only predictions.

Never runs Torch, EPD, Gazebo, ROS, or EMD. Does not modify a prediction,
mask, box, confidence, association threshold, source evidence or model.
The renderer identities are diagnostic truth only. No corrective mask is output.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASES = (
    ("#221", ROOT / "evidence/segmentation_v2/mask_finetune_001",
     Path("/home/user/epd-v2-artifacts/mask-finetune-stage-a2-001")),
    ("#222", ROOT / "evidence/segmentation_v2/boundary_mask_001",
     Path("/home/user/epd-v2-artifacts/boundary-mask-stage-a2-001")),
)
EXPECTED_CANDIDATE_SHA = "bf3ac58116a3a692806ed20b25fb5414277e76da605cf0f60e2e5a23656bfbfe"
EXPECTED_STAGE_A2_STATUS = "BLOCKED_STRICT_ASSOCIATION"
EXPECTED_RGB_SHA = "92f1ba8ad58b2437fa5ccf55e73d53cc9dcba469b04efefbe624ce1505585793"
ID256 = "capture.json.ids.u32"
ID512 = "capture.json.native512.ids.u32"
RGB = "capture.json.rgb8"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_predictions(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != {"boxes", "labels", "scores", "masks"}:
            raise ValueError("PREDICTION_CONTRACT: wrong keys")
        pred = {key: np.array(bundle[key], copy=True) for key in bundle.files}
    n = pred["scores"].shape[0]
    spec = {
        "boxes": ((n, 4), np.dtype("float32")),
        "labels": ((n,), np.dtype("int64")),
        "scores": ((n,), np.dtype("float32")),
        "masks": ((n, 1, 512, 512), np.dtype("float32")),
    }
    for key, (shape, dtype) in spec.items():
        if pred[key].shape != shape or pred[key].dtype != dtype or not np.isfinite(pred[key]).all():
            raise ValueError("PREDICTION_CONTRACT: invalid " + key)
    if ((pred["scores"] < 0) | (pred["scores"] > 1)).any() or (
            (pred["masks"] < 0) | (pred["masks"] > 1)).any():
        raise ValueError("PREDICTION_CONTRACT: probabilities out of range")
    if (pred["labels"] != 1).any() or (pred["boxes"] < 0).any() or (pred["boxes"] > 512).any():
        raise ValueError("PREDICTION_CONTRACT: class or box range")
    if (pred["boxes"][:, 2:] <= pred["boxes"][:, :2]).any():
        raise ValueError("PREDICTION_CONTRACT: degenerate box")
    return pred


def bounds(box: np.ndarray) -> tuple[int, int, int, int]:
    # Match retained Stage-A2 tester: float boxes integer-truncated after clipping.
    x0, y0, x1, y1 = (int(v) for v in np.clip(box, 0, 512))
    if not (0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512):
        raise ValueError("PREDICTION_CONTRACT: invalid integer ROI")
    return x0, y0, x1, y1


def project(prediction: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unmodified Stage-A2 contract: ROI>0.5 then conservative union of four samples."""
    if prediction.shape != (512, 512) or prediction.dtype != np.float32:
        raise ValueError("PREDICTION_CONTRACT: mask shape")
    x0, y0, x1, y1 = bounds(box)
    whole = np.zeros((512, 512), dtype=bool)
    whole[y0:y1, x0:x1] = prediction[y0:y1, x0:x1] > .5
    return whole, whole.reshape(256, 2, 256, 2).any(axis=(1, 3))


def raw_capture(capture: Path, name: str, hashes: dict) -> bytes:
    if name not in hashes:
        raise ValueError("CAPTURE_IDENTITY: missing " + name)
    path = capture / name
    if path.is_file():
        data = path.read_bytes()
    elif (capture / (name + ".gz")).is_file():
        with gzip.open(capture / (name + ".gz"), "rb") as stream:
            data = stream.read()
    else:
        raise ValueError("CAPTURE_IDENTITY: missing bytes for " + name)
    if digest_bytes(data) != hashes[name]:
        raise ValueError("CAPTURE_IDENTITY: SHA mismatch " + name)
    return data


def read_capture(stage: Path, report: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    capture = stage / "capture"
    manifest = capture / "raw_sha256.json"
    if digest(manifest) != report["capture_manifest_sha256"]:
        raise ValueError("CAPTURE_IDENTITY: checked manifest changed")
    hashes = read_json(manifest)
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("CAPTURE_IDENTITY: empty manifest")
    # Verify every retained capture artifact; not just the two ID images.
    for name in sorted(hashes):
        raw_capture(capture, name, hashes)
    rgb = raw_capture(capture, RGB, hashes)
    if digest_bytes(rgb) != report["input_rgb_sha256"] or digest_bytes(rgb) != EXPECTED_RGB_SHA:
        raise ValueError("CAPTURE_IDENTITY: different Stage-A2 RGB")
    a = np.frombuffer(raw_capture(capture, ID256, hashes), dtype="<u4")
    b = np.frombuffer(raw_capture(capture, ID512, hashes), dtype="<u4")
    if a.size != 256*256 or b.size != 512*512:
        raise ValueError("CAPTURE_IDENTITY: invalid native ID raster shapes")
    return a.reshape(256, 256), b.reshape(512, 512), hashes


def validate_recorded(stage: Path, report: dict, pred: dict, *, expected_checkpoint_sha: str | None) -> None:
    if report.get("status") != EXPECTED_STAGE_A2_STATUS or report.get("association", {}).get("accepted") != 0:
        raise ValueError("STAGE_A2_RESULT_DRIFT: expected retained strict rejection 0/2")
    chosen = report.get("checkpoint", {})
    if expected_checkpoint_sha is not None and chosen.get("sha256") != expected_checkpoint_sha:
        raise ValueError("STAGE_A2_RESULT_DRIFT: wrong 56-grid checkpoint")
    process = read_json(stage / "process.json")
    if (process.get("exit_code") != 2 or process.get("resource_limit_reason") is not None or
            len(pred["scores"]) != len(report.get("measurements", [])) or
            len(pred["scores"]) != len(report["association"]["detections"])):
        raise ValueError("STAGE_A2_RESULT_DRIFT: process or detection inventory mismatched")
    for i, (measurement, detection) in enumerate(zip(
            report["measurements"], report["association"]["detections"])):
        if measurement.get("prediction_index") != i or detection.get("prediction_index") != i:
            raise ValueError("STAGE_A2_RESULT_DRIFT: detection order changed")
        if (abs(float(pred["scores"][i]) - measurement["score"]) > 1e-7 or
                abs(float(pred["scores"][i]) - detection["confidence"]) > 1e-7 or
                list(bounds(pred["boxes"][i])) != detection["bbox"]):
            raise ValueError("STAGE_A2_RESULT_DRIFT: recorded boxes/scores differ")
    # No arbitrary user-provided result can cause diagnostic promotion.


def histogram(dist: np.ndarray) -> dict:
    if not dist.size:
        return {"pixels": 0, "distance_le_1px": 0, "distance_1_to_2px": 0,
                "distance_2_to_3px": 0, "distance_gt_3px": 0,
                "max_distance_px": None, "p95_distance_px": None}
    return {
        "pixels": int(dist.size),
        "distance_le_1px": int((dist <= 1).sum()),
        "distance_1_to_2px": int(((dist > 1) & (dist <= 2)).sum()),
        "distance_2_to_3px": int(((dist > 2) & (dist <= 3)).sum()),
        "distance_gt_3px": int((dist > 3).sum()),
        "max_distance_px": float(np.max(dist)),
        "p95_distance_px": float(np.percentile(dist, 95)),
    }


def instance(pred: dict, i: int, identity: int, ids256: np.ndarray,
             ids512: np.ndarray) -> tuple[dict, dict]:
    """Measured masks and optional coordinate arrays, no prediction modification."""
    prob = pred["masks"][i, 0]
    roi512, mapped256 = project(prob, pred["boxes"][i])
    raw512 = prob > .5
    native_object = ids512 == identity
    mapped_object = ids256 == identity
    if not native_object.any() or not mapped_object.any():
        raise ValueError("CAPTURE_IDENTITY: reference object absent")
    full_background = raw512 & (ids512 == 0)
    roi_background = roi512 & (ids512 == 0)
    mapped_background = mapped256 & (ids256 == 0)
    native_cross = roi512 & (ids512 != 0) & ~native_object
    mapped_cross = mapped256 & (ids256 != 0) & ~mapped_object
    native_missing = native_object & ~raw512
    mapped_missing = mapped_object & ~mapped256
    d512 = cv2.distanceTransform((~native_object).astype(np.uint8), cv2.DIST_L2,
                                 cv2.DIST_MASK_PRECISE)
    d256 = cv2.distanceTransform((~mapped_object).astype(np.uint8), cv2.DIST_L2,
                                 cv2.DIST_MASK_PRECISE)
    ys, xs = np.nonzero(roi_background)
    ym, xm = np.nonzero(mapped_background)
    native_probs = prob[ys, xs]
    # Native512 vs independently rendered256 cells; mismatch is descriptive,
    # not a decomposition of learned-mask versus resizing cause.
    raster_disagreement = ids512 != ids256.repeat(2, axis=0).repeat(2, axis=1)
    outcome = {
        "prediction_index": i, "reference_identity": identity,
        "confidence": float(pred["scores"][i]),
        "box": list(bounds(pred["boxes"][i])),
        "native512_raw_background": int(full_background.sum()),
        "native512_roi_background": int(roi_background.sum()),
        "native512_raw_missing": int(native_missing.sum()),
        "native512_roi_cross_object": int(native_cross.sum()),
        "mapped256_background": int(mapped_background.sum()),
        "mapped256_missing": int(mapped_missing.sum()),
        "mapped256_cross_object": int(mapped_cross.sum()),
        "native512_background_removed_by_bbox": int(full_background.sum()-roi_background.sum()),
        "native512_background_distance": histogram(d512[roi_background]),
        "mapped256_background_distance": histogram(d256[mapped_background]),
        "native512_background_probabilities": {
            "above_0_9": int((native_probs > .9).sum()),
            "median": float(np.median(native_probs)) if native_probs.size else None,
        },
        "native512_id_disagreement_with_repeated256": int(raster_disagreement.sum()),
        "native512_id_disagreement_with_repeated256_inside_roi_foreground":
            int((raster_disagreement & roi512).sum()),
        "strict_association_authority": False,
    }
    arrays = {
        f"d{i}_native512_background_yx": np.column_stack((ys, xs)).astype(np.int32),
        f"d{i}_native512_background_prob": native_probs.astype(np.float32),
        f"d{i}_native512_background_distance_px": d512[ys, xs].astype(np.float32),
        f"d{i}_mapped256_background_yx": np.column_stack((ym, xm)).astype(np.int32),
        f"d{i}_mapped256_background_distance_px": d256[ym, xm].astype(np.float32),
    }
    return outcome, arrays


def check_instance_against_stage(measured: dict, recorded: dict, ids256: np.ndarray,
                                 pred: dict, idx: int) -> None:
    identity = measured["reference_identity"]
    _, mapped256 = project(pred["masks"][idx, 0], pred["boxes"][idx])
    counts = {str(int(value)): int(n) for value, n in zip(
        *np.unique(ids256[mapped256], return_counts=True))}
    for field in ("native512_raw_background", "native512_roi_background",
                  "mapped_missing_pixels", "native512_raw_missing"):
        equivalent = "mapped256_missing" if field == "mapped_missing_pixels" else field
        if measured[equivalent] != recorded[field]:
            raise ValueError("STAGE_A2_RECOMPUTE_MISMATCH: " + field)
    if counts != recorded["mapped_id_counts"]:
        raise ValueError("STAGE_A2_RECOMPUTE_MISMATCH: mapped ID inventory")
    if measured["native512_roi_cross_object"] or measured["mapped256_cross_object"]:
        raise ValueError("STAGE_A2_RECOMPUTE_MISMATCH: unexpected cross-object pixels")


def baseline(name: str, evidence: Path, external: Path, ids256: np.ndarray,
             ids512: np.ndarray, current: dict, current_predictions: dict) -> dict:
    manifest_path = evidence / "artifact_locations.json"
    manifest = read_json(manifest_path)
    rows = manifest["files"] if isinstance(manifest, dict) else manifest
    index = {item["path"]: item["sha256"] for item in rows}
    retained = evidence / "stage_a2_result.json"
    old = read_json(retained)
    expected_result = index.get(str(external / "result.json"))
    if not expected_result or digest(retained) != expected_result:
        raise ValueError("REFERENCE_IDENTITY: archived " + name + " result hash differs")
    if (old["input_rgb_sha256"] != current["input_rgb_sha256"] or
            old["capture_manifest_sha256"] != current["capture_manifest_sha256"] or
            old.get("association", {}).get("accepted") != 0):
        raise ValueError("REFERENCE_IDENTITY: incomparable capture or association")
    summaries = []
    for i in (0, 1):
        measurement = old["measurements"][i]
        if measurement["reference_identity"] != current["measurements"][i]["reference_identity"]:
            raise ValueError("REFERENCE_IDENTITY: cube order differs")
        summaries.append({
            "prediction_index": i,
            "native512_raw_background": measurement["native512_raw_background"],
            "native512_roi_background": measurement["native512_roi_background"],
            "mapped256_background": int(measurement["mapped_id_counts"].get("0", 0)),
            "mapped256_missing": measurement["mapped_missing_pixels"],
        })
    report = {
        "model": name,
        "capture_identity_verified": True,
        "result_sha256": digest(retained),
        "status": old["status"], "strict_accepted": old["association"]["accepted"],
        "summaries": summaries, "pixelwise_overlap": "UNAVAILABLE",
        "pixelwise_details": None,
    }
    pred_path = external / "predictions.npz"
    expected_pred = index.get(str(pred_path))
    if not expected_pred or not pred_path.exists():
        return report
    if digest(pred_path) != expected_pred:
        raise ValueError("REFERENCE_IDENTITY: changed archived " + name + " predictions")
    pred = load_predictions(pred_path)
    if len(pred["scores"]) != len(old["measurements"]):
        raise ValueError("REFERENCE_IDENTITY: prediction count mismatch")
    overlap = []
    for i in (0, 1):
        identity = old["measurements"][i]["reference_identity"]
        prior, _ = instance(pred, i, identity, ids256, ids512)
        if (prior["native512_raw_background"] != summaries[i]["native512_raw_background"] or
                prior["native512_roi_background"] != summaries[i]["native512_roi_background"] or
                prior["mapped256_background"] != summaries[i]["mapped256_background"]):
            raise ValueError("REFERENCE_IDENTITY: archived prediction metrics disagree")
        new_roi, new_map = project(current_predictions["masks"][i,0], current_predictions["boxes"][i])
        old_roi, old_map = project(pred["masks"][i,0], pred["boxes"][i])
        b512 = ids512 == 0
        b256 = ids256 == 0
        both512 = int((new_roi & old_roi & b512).sum())
        both256 = int((new_map & old_map & b256).sum())
        overlap.append({
            "prediction_index": i,
            "native512_shared_spill": both512,
            "native512_removed_from_prior": int((old_roi & b512).sum())-both512,
            "native512_added_vs_prior": int((new_roi & b512).sum())-both512,
            "mapped256_shared_spill": both256,
            "mapped256_removed_from_prior": int((old_map & b256).sum())-both256,
            "mapped256_added_vs_prior": int((new_map & b256).sum())-both256,
        })
    report["pixelwise_overlap"] = "VERIFIED_ARCHIVED_RAW_PREDICTIONS"
    report["pixelwise_details"] = overlap
    return report


def audit(stage: Path, campaign: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("OUTPUT_NOT_EXCLUSIVE: keep original evidence intact")
    stage = stage.resolve(strict=True)
    campaign = campaign.resolve(strict=True)
    if not stage.is_dir() or not campaign.is_dir():
        raise ValueError("INPUT_NOT_DIRECTORY")
    report = read_json(stage / "result.json")
    pred_path = stage / "predictions.npz"
    pred = load_predictions(pred_path)
    validate_recorded(stage, report, pred, expected_checkpoint_sha=EXPECTED_CANDIDATE_SHA)
    run = read_json(campaign / "result.json")
    chosen = run.get("selected") or {}
    if (run.get("status") != "VALIDATION_PASS" or run.get("steps") != 300 or
            run.get("epochs") != 1 or run.get("architecture") != "mask_roi28_logits56" or
            chosen.get("sha256") != EXPECTED_CANDIDATE_SHA or
            digest(Path(chosen["path"])) != EXPECTED_CANDIDATE_SHA or
            run.get("held_out_files_opened") != 0 or run.get("stage_a2_used_for_selection") is not False):
        raise ValueError("TRAINING_IDENTITY: candidate input or validation status drift")
    if report["checkpoint"] != chosen:
        raise ValueError("TRAINING_IDENTITY: Stage-A2 checkpoint does not match selection")
    hashes = read_json(campaign / "binding.json")
    if (hashes.get("checkpoint") != "f3e7531c25cb20920c04283d205d7241028d7321b1ea34e2f1848415744a3b56" or
            hashes.get("dataset") != "04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11" or
            hashes.get("policy", {}).get("schema") != "epd_56_grid_one_epoch/v1"):
        raise ValueError("TRAINING_IDENTITY: original checkpoint, data or policy binding drift")
    if Path(chosen["path"]).resolve() != (campaign / "epoch_01.pt").resolve():
        raise ValueError("TRAINING_IDENTITY: candidate is outside recorded campaign directory")
    history = [json.loads(line) for line in (campaign / "history.jsonl").read_text().splitlines()]
    if (len(history) != 300 or
            [entry.get("step") for entry in history] != list(range(1, 301)) or
            len({entry.get("scene_id") for entry in history}) != 300 or
            any(entry.get("epoch") != 1 for entry in history)):
        raise ValueError("TRAINING_IDENTITY: incomplete 300-update history")
    ids256, ids512, _ = read_capture(stage, report)
    scores = report["association"]["detections"]
    confident = [i for i, row in enumerate(scores) if row["confidence"] >= .8]
    if confident != [0, 1]:
        raise ValueError("STAGE_A2_RESULT_DRIFT: expected exactly two confident cubes at 0 and 1")
    measures = []
    coords = {}
    for i in confident:
        measured, arrays = instance(pred, i, report["measurements"][i]["reference_identity"], ids256, ids512)
        check_instance_against_stage(measured, report["measurements"][i], ids256, pred, i)
        measures.append(measured)
        coords.update(arrays)
    previous = [baseline(name, evidence, external, ids256, ids512, report, pred)
                for name, evidence, external in BASES]
    process_path = campaign / "process.json"
    if process_path.is_file():
        training_process = read_json(process_path)
        supervisor = ("PRESENT_EXIT_0" if training_process.get("exit_code") == 0 and
                      training_process.get("resource_limit_reason") is None
                      else "PRESENT_NOT_QUALIFIED")
    else:
        supervisor = "MISSING_UNRESOLVED"
    summary = {
        "schema": "epd_stage_a2_readonly_boundary_audit/v1",
        "decision": "DIAGNOSTIC_ONLY_BLOCKED_NO_PROMOTION",
        "source_model": "56-grid epoch1",
        "checkpoint_sha256": EXPECTED_CANDIDATE_SHA,
        "selected_validation_pass": True,
        "stage_a2_status": report["status"],
        "strict_accepted": report["association"]["accepted"],
        "training_supervisor_evidence": supervisor,
        "missing_supervisor_does_not_fabricate_success": True,
        "capture_manifest_sha256": report["capture_manifest_sha256"],
        "rgb_sha256": report["input_rgb_sha256"],
        "inputs_sha256": {
            "stage_a2_result": digest(stage / "result.json"),
            "stage_a2_process": digest(stage / "process.json"),
            "stage_a2_predictions": digest(pred_path),
            "campaign_result": digest(campaign / "result.json"),
            "campaign_checkpoint": EXPECTED_CANDIDATE_SHA,
            "capture_manifest": digest(stage / "capture/raw_sha256.json"),
            "audit_source": digest(Path(__file__)),
        },
        "confidence_threshold": .8, "mask_threshold_strict_gt": .5,
        "current": measures,
        "previous": previous,
        "limitations": {
            "mask_probabilities": "Full-image post-paste probabilities, not isolated learned ROI-head logits",
            "native_vs_mapped": "512 IDs and 256 IDs are separately rendered; repeating 256 cells is not native 512 truth",
            "causal_attribution": "UNKNOWN: learned mask, ROI box/paste, interpolation and native raster error cannot be uniquely partitioned from these artifacts",
            "counterfactual_mask_correction": "NOT_PERFORMED",
            "simulator_or_onnx_inference": "NOT_PERFORMED",
            "held_out_pixels_accessed": False,
            "workcell_motion": False,
            "physical_contact_qualified": False,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    coordinates = output / "boundary_coordinates.npz"
    np.savez_compressed(coordinates, **coords)
    summary["diagnostic_coordinates_sha256"] = digest(coordinates)
    (output / "boundary_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": summary["decision"], "strict_accepted": summary["strict_accepted"],
        "training_supervisor_evidence": supervisor,
        "native_roi_background": [m["native512_roi_background"] for m in measures],
        "mapped_background": [m["mapped256_background"] for m in measures],
        "baseline_pixel_overlap": {b["model"]: b["pixelwise_overlap"] for b in previous},
        "report": str(output / "boundary_report.json"),
    }, sort_keys=True))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-a2", type=Path, default=Path(
        "/home/user/epd-v2-artifacts/mask-resolution-stage-a2-001"))
    ap.add_argument("--campaign", type=Path, default=Path(
        "/home/user/epd-v2-artifacts/mask-resolution-campaign-001"))
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    try:
        audit(args.stage_a2, args.campaign, args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
