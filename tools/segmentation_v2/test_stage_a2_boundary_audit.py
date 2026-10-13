"""Pure offline boundary-audit regressions; no model, inference, ROS or captures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import stage_a2_boundary_audit as a


class StageA2BoundaryAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.ids256 = np.zeros((256, 256), dtype=np.uint32)
        self.ids512 = np.zeros((512, 512), dtype=np.uint32)
        self.ids256[2:4, 2:4] = 65517
        self.ids512[4:8, 4:8] = 65517
        self.prob = np.zeros((512, 512), dtype=np.float32)
        self.prob[4:8, 4:8] = .99
        self.prob[3, 4] = .95  # One actual native background pixel.
        self.box = np.array([0., 0., 512., 512.], dtype=np.float32)
        self.pred = {
            "boxes": self.box.reshape(1,4),
            "labels": np.array([1], dtype=np.int64),
            "scores": np.array([.98], dtype=np.float32),
            "masks": self.prob.reshape(1,1,512,512),
        }

    def test_strict_threshold_and_conservative_union_no_correction(self):
        self.prob[3,4] = .5
        self.prob[4,4] = .5
        native, mapped = a.project(self.prob, self.box)
        self.assertFalse(native[3,4])
        self.assertFalse(native[4,4])
        self.assertTrue(native[4,5])
        self.assertTrue(mapped[2,2])
        self.assertFalse(mapped[1,2])  # strict greater-than threshold
        self.assertEqual(float(self.prob[3,4]), .5)  # input unchanged

    def test_native_mapped_background_and_distance(self):
        measured, coords = a.instance(self.pred, 0, 65517, self.ids256, self.ids512)
        self.assertEqual(measured["native512_raw_background"], 1)
        self.assertEqual(measured["native512_roi_background"], 1)
        self.assertEqual(measured["mapped256_background"], 1)
        self.assertEqual(measured["mapped256_missing"], 0)
        self.assertEqual(measured["native512_raw_missing"], 0)
        self.assertEqual(measured["native512_background_distance"]["distance_le_1px"], 1)
        self.assertEqual(measured["native512_background_probabilities"]["above_0_9"], 1)
        np.testing.assert_array_equal(coords["d0_native512_background_yx"], [[3,4]])
        np.testing.assert_array_equal(coords["d0_mapped256_background_yx"], [[1,2]])
        self.assertEqual(measured["native512_id_disagreement_with_repeated256"], 0)

    def test_stage_a2_count_reconciliation_and_tamper(self):
        measurement, _ = a.instance(self.pred, 0, 65517, self.ids256, self.ids512)
        row = {
            "native512_raw_background": 1, "native512_roi_background": 1,
            "native512_raw_missing": 0, "mapped_missing_pixels": 0,
            "mapped_id_counts": {"0": 1, "65517": 4},
        }
        a.check_instance_against_stage(measurement, row, self.ids256, self.pred, 0)
        changed = dict(row, mapped_id_counts={"0": 0, "65517": 4})
        with self.assertRaisesRegex(ValueError, "RECOMPUTE_MISMATCH"):
            a.check_instance_against_stage(measurement, changed, self.ids256, self.pred, 0)

    def test_crop_change_measured_not_relabelled(self):
        clipped = self.pred["boxes"].copy()
        clipped[0] = [4,4,8,8]
        pred = dict(self.pred, boxes=clipped)
        measured, _ = a.instance(pred, 0, 65517, self.ids256, self.ids512)
        self.assertEqual(measured["native512_raw_background"], 1)
        self.assertEqual(measured["native512_roi_background"], 0)
        self.assertEqual(measured["native512_background_removed_by_bbox"], 1)
        self.assertFalse(measured["strict_association_authority"])
        self.assertEqual(self.prob[3,4], np.float32(.95))

    def test_invalid_probability_fails_closed(self):
        path = self.folder / "invalid.npz"
        pred = dict(self.pred, masks=self.pred["masks"].copy())
        pred["masks"][0,0,0,0] = np.nan
        np.savez_compressed(path, **pred)
        with self.assertRaisesRegex(ValueError, "PREDICTION_CONTRACT"):
            a.load_predictions(path)
        self.assertFalse((self.folder / "boundary_report.json").exists())

    def test_result_requires_original_order_and_confidence(self):
        stage = self.folder / "stage"
        stage.mkdir()
        (stage / "process.json").write_text(json.dumps({
            "exit_code": 2, "resource_limit_reason": None}))
        report = {
            "status": "BLOCKED_STRICT_ASSOCIATION",
            "association": {"accepted": 0, "detections": [
                {"prediction_index": 0, "bbox": [0,0,512,512],
                 "confidence": float(self.pred["scores"][0])}]},
            "measurements": [{"prediction_index": 0, "score": float(self.pred["scores"][0])}],
            "checkpoint": {"sha256": "correct"},
        }
        a.validate_recorded(stage, report, self.pred, expected_checkpoint_sha="correct")
        report["association"]["detections"][0]["confidence"] = .7
        with self.assertRaisesRegex(ValueError, "STAGE_A2_RESULT_DRIFT"):
            a.validate_recorded(stage, report, self.pred, expected_checkpoint_sha="correct")
        report["association"]["detections"][0]["confidence"] = float(self.pred["scores"][0])
        (stage / "process.json").write_text(json.dumps({
            "exit_code": 0, "resource_limit_reason": None}))
        with self.assertRaisesRegex(ValueError, "STAGE_A2_RESULT_DRIFT"):
            a.validate_recorded(stage, report, self.pred, expected_checkpoint_sha="correct")

    def test_archived_summary_fallback_when_raw_archive_absent(self):
        evidence = self.folder / "reference"
        evidence.mkdir()
        external = self.folder / "offline-prior"
        old = {
            "status": "BLOCKED_STRICT_ASSOCIATION",
            "input_rgb_sha256": "same RGB",
            "capture_manifest_sha256": "same manifest",
            "association": {"accepted": 0},
            "measurements": [
                {"native512_raw_background": 10,
                 "native512_roi_background": 8,
                 "mapped_id_counts": {"0": 3, "65517": 4}},
                {"native512_raw_background": 12,
                 "native512_roi_background": 11,
                 "mapped_id_counts": {"0": 7, "65523": 5}},
            ],
        }
        result = evidence / "stage_a2_result.json"
        result.write_text(json.dumps(old))
        (evidence / "artifact_locations.json").write_text(json.dumps([
            {"path": str(external / "result.json"), "sha256": a.digest(result)},
            {"path": str(external / "predictions.npz"), "sha256": "f"*64},
        ]))
        current = {
            "input_rgb_sha256": "same RGB",
            "capture_manifest_sha256": "same manifest",
            "measurements": [{"reference_identity": 65517},
                             {"reference_identity": 65523}],
        }
        with patch.object(a, "BASES", ()):  # no access to actual external evidence
            base = a.baseline("#221", evidence, external, self.ids256, self.ids512,
                              current, self.pred)
        self.assertEqual(base["pixelwise_overlap"], "UNAVAILABLE")
        self.assertEqual(base["summaries"][0]["mapped256_background"], 3)
        self.assertEqual(base["summaries"][0]["reference_identity"], 65517)
        self.assertEqual(base["summaries"][0]["reference_identity_provenance"],
                         "DERIVED_FROM_UNAMBIGUOUS_RECORDED_ID_COUNTS")
        self.assertIsNone(base["summaries"][0]["mapped256_missing"])
        self.assertIn("mapped_missing_pixels",
                      base["summaries"][0]["unrecorded_fields_remain_unknown"])
        self.assertIsNone(base["pixelwise_details"])

    def test_missing_archive_hash_never_claims_pixel_overlap(self):
        evidence = self.folder / "reference"
        evidence.mkdir()
        external = self.folder / "offline-prior"
        old = {
            "status": "BLOCKED_STRICT_ASSOCIATION",
            "input_rgb_sha256": "same RGB",
            "capture_manifest_sha256": "same manifest",
            "association": {"accepted": 0},
            "measurements": [{"reference_identity": 65517,
                              "native512_raw_background": 0, "native512_roi_background": 0,
                              "mapped_id_counts": {"65517": 4}, "mapped_missing_pixels": 0},
                             {"reference_identity": 65523,
                              "native512_raw_background": 0, "native512_roi_background": 0,
                              "mapped_id_counts": {"65523": 5}, "mapped_missing_pixels": 0}],
        }
        result = evidence / "stage_a2_result.json"
        result.write_text(json.dumps(old))
        external.mkdir()
        (external / "predictions.npz").write_bytes(b"tampered archive")
        (evidence / "artifact_locations.json").write_text(json.dumps([
            {"path": str(external / "result.json"), "sha256": a.digest(result)},
            {"path": str(external / "predictions.npz"), "sha256": "f"*64},
        ]))
        current = {"input_rgb_sha256": "same RGB",
                   "capture_manifest_sha256": "same manifest",
                   "measurements": [{"reference_identity": 65517},
                                    {"reference_identity": 65523}]}
        with self.assertRaisesRegex(ValueError, "changed archived"):
            a.baseline("#221", evidence, external, self.ids256, self.ids512,
                       current, self.pred)

    def test_actual_committed_221_legacy_schema_is_supported(self):
        # #221 was recorded by an earlier Stage-A2 checker that stored
        # mapped_id_counts but neither reference_identity nor missing pixels.
        source = a.ROOT / "evidence/segmentation_v2/mask_finetune_001/stage_a2_result.json"
        historic = json.loads(source.read_text())
        first, second = historic["measurements"][:2]
        self.assertNotIn("reference_identity", first)
        self.assertNotIn("mapped_missing_pixels", first)
        self.assertEqual(first["mapped_id_counts"]["65517"], 2079)
        self.assertEqual(second["mapped_id_counts"]["65523"], 2166)
        self.assertEqual(a.historic_reference_identity(first, 65517, "#221"),
                         (65517, "DERIVED_FROM_UNAMBIGUOUS_RECORDED_ID_COUNTS"))
        self.assertEqual(a.historic_reference_identity(second, 65523, "#221")[0], 65523)

    def test_legacy_identity_inference_rejects_ambiguity_or_drift(self):
        counts = {"mapped_id_counts": {"0": 1, "65517": 4}}
        self.assertEqual(a.historic_reference_identity(counts, 65517, "#221")[0], 65517)
        with self.assertRaisesRegex(ValueError, "ambiguous or different"):
            a.historic_reference_identity(counts, 65523, "#221")
        with self.assertRaisesRegex(ValueError, "ambiguous or different"):
            a.historic_reference_identity(
                {"mapped_id_counts": {"0": 1, "65517": 4, "65523": 1}}, 65517, "#221")
        with self.assertRaisesRegex(ValueError, "recorded reference identity disagrees"):
            a.historic_reference_identity(
                {"mapped_id_counts": {"65517": 4}, "reference_identity": 65523},
                65517, "#222")
        with self.assertRaisesRegex(ValueError, "missing archived"):
            a.historic_reference_identity({}, 65517, "#221")

    def test_distance_statistics_empty_and_outlier(self):
        empty = a.histogram(np.array([], dtype=np.float32))
        self.assertEqual(empty["pixels"], 0)
        self.assertIsNone(empty["max_distance_px"])
        d = a.histogram(np.array([1., 1.5, 2.5, 7.], dtype=np.float32))
        self.assertEqual([d[k] for k in ("distance_le_1px", "distance_1_to_2px",
                                        "distance_2_to_3px", "distance_gt_3px")],
                         [1,1,1,1])

    def test_existing_output_is_never_overwritten(self):
        out = self.folder / "report"
        out.mkdir()
        keep = out / "important.json"
        keep.write_text("retain")
        with self.assertRaisesRegex(ValueError, "OUTPUT_NOT_EXCLUSIVE"):
            a.audit(self.folder / "no-stage", self.folder / "no-campaign", out)
        self.assertEqual(keep.read_text(), "retain")


if __name__ == "__main__":
    unittest.main()
