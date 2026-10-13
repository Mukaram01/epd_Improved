"""Fast offline policy/resource/identity gates; DO NOT run 300-update training."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pilot import sha
import mask_resolution_campaign as c


class FiftySixGridCampaignTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "initial.pt"
        self.dataset = self.root / "dataset.json"
        self.checkpoint.write_bytes(b"frozen selected checkpoint")
        self.dataset.write_bytes(b"frozen dataset manifest")
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.output = self.root / "fresh-run"
        self.evidence_paths = {
            "failed_process.json": {
                "exit_code": 1, "resource_limit_reason": None,
                "wall_seconds": 17.902557232999243, "child_peak_rss_kib": 2062980,
            },
            "unattempted_process.json": {
                "exit_code": 0, "resource_limit_reason": None,
                "child_peak_rss_kib": 1716808,
            },
            "unattempted_result.json": {
                "status": "FEASIBILITY_PASS",
                "source_sha256": sha(c.HERE / "mask_resolution.py"),
                "initialization_sha256": sha(self.checkpoint),
                "dataset_sha256": sha(self.dataset),
                "benchmark_updates": 4, "training_epochs": 0,
                "campaign_updates": 0, "accuracy_evaluated": False,
                "qualified_model": False, "stage_a2_inference": False,
                "measurements": [
                    {"scene_id": "expansion_093", "seconds": 6.606864817003952},
                    {"scene_id": "expansion_123", "seconds": 4.050797860996681},
                ],
                "step_time_upper_bound": 17.902557232999243,
                "projected_one_epoch_seconds": (300 + 24) * 17.902557232999243 * 1.5 + 240,
                "projected_two_epoch_seconds": (600 + 36) * 17.902557232999243 * 1.5 + 240,
                "projected_rss_kib": 2062980 * 1.5,
            },
        }
        self._write()
        for name, value in (
                ("EVIDENCE", self.evidence), ("INITIAL_SHA", sha(self.checkpoint)),
                ("DATASET_SHA", sha(self.dataset))):
            active = patch.object(c, name, value)
            active.start()
            self.addCleanup(active.stop)
        disk = patch.object(c.shutil, "disk_usage",
                            return_value=SimpleNamespace(free=80 * 1024**3))
        disk.start()
        self.addCleanup(disk.stop)
        cores = patch.object(c.os, "cpu_count", return_value=8)
        cores.start()
        self.addCleanup(cores.stop)

    def _write(self):
        for name, value in self.evidence_paths.items():
            (self.evidence / name).write_text(json.dumps(value))

    def test_exact_one_epoch_policy_and_frozen_metric_gates(self):
        self.assertEqual(c.POLICY["epochs"], 1)
        self.assertEqual(c.POLICY["training_samples"], 300)
        self.assertEqual(c.POLICY["validation_samples"], 12)
        self.assertEqual(c.POLICY["mask_roi_output_size"], 28)
        self.assertEqual(c.POLICY["mask_logits_size"], 56)
        self.assertEqual(c.POLICY["gate"]["minimum_recovered"], 18)
        self.assertEqual(c.POLICY["gate"]["background_spill_strict_lt"], 1915)
        self.assertEqual(c.POLICY["gate"]["confidence_threshold"], 0.8)
        self.assertEqual(c.POLICY["gate"]["mask_threshold_strict_gt"], 0.5)
        self.assertIs(c.POLICY["automatic_retry"], False)

    def test_valid_retained_feasibility_admits_one_not_two_epochs(self):
        admitted = c.evidence_admission(
            checkpoint=self.checkpoint, artifacts=self.dataset, output=self.output)
        self.assertEqual(admitted["status"], "GO_ONE_EPOCH_ONLY")
        self.assertEqual(admitted["training_updates_authorized"], 300)
        self.assertLess(admitted["projected_seconds"], 10800)
        self.assertEqual(len(admitted["source_evidence_sha256"]), 3)
        self.assertIs(admitted["model_qualified"], False)
        self.assertFalse(self.output.exists(), "Check must not create a campaign output")
        self.assertEqual(admitted["checkpoint_sha256"], sha(self.checkpoint))

    def test_no_second_run_or_artifact_overwrite(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "OUTPUT_NOT_EXCLUSIVE"):
            c.evidence_admission(
                checkpoint=self.checkpoint, artifacts=self.dataset, output=self.output)

    def test_checkpoint_and_dataset_hashes_are_strict(self):
        self.checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "TRAINING_INPUT_DRIFT"):
            c.evidence_admission(
                checkpoint=self.checkpoint, artifacts=self.dataset, output=self.output)
        self.checkpoint.write_bytes(b"frozen selected checkpoint")
        self.dataset.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "TRAINING_INPUT_DRIFT"):
            c.evidence_admission(
                checkpoint=self.checkpoint, artifacts=self.dataset, output=self.output)

    def test_prior_process_failure_must_not_be_a_resource_error(self):
        self.evidence_paths["failed_process.json"]["resource_limit_reason"] = "wall_timeout"
        self._write()
        with self.assertRaisesRegex(ValueError, "FEASIBILITY_EVIDENCE_INVALID"):
            c.evidence_admission(
                checkpoint=self.checkpoint, artifacts=self.dataset, output=self.output)

    def test_rejects_unbounded_resource_prediction(self):
        result = self.evidence_paths["unattempted_result.json"]
        result["step_time_upper_bound"] = 30.
        result["projected_one_epoch_seconds"] = (300 + 24) * 30. * 1.5 + 240
        self._write()
        with self.assertRaisesRegex(ValueError, "RESOURCE_ADMISSION_FAILED"):
            c.evidence_admission(
                checkpoint=self.checkpoint, artifacts=self.dataset, output=self.output)

    def test_refuses_silent_benchmark_or_architecture_source_drift(self):
        self.evidence_paths["unattempted_result.json"]["source_sha256"] = "0" * 64
        self._write()
        with self.assertRaisesRegex(ValueError, "FEASIBILITY_EVIDENCE_INVALID"):
            c.evidence_admission(
                checkpoint=self.checkpoint, artifacts=self.dataset, output=self.output)

    def test_disk_budget_requires_five_gib_reserve_plus_output_allowance(self):
        with patch.object(c.shutil, "disk_usage",
                          return_value=SimpleNamespace(free=5 * 1024**3)):
            with self.assertRaisesRegex(ValueError, "RESOURCE_ADMISSION_FAILED"):
                c.evidence_admission(
                    checkpoint=self.checkpoint, artifacts=self.dataset, output=self.output)

    def test_strict_stage_a2_checker_reconstructs_declared_architecture(self):
        text = (c.HERE / "stage_a2_mask_check.py").read_text()
        self.assertIn("model=configure56(make_model())", text)
        self.assertIn("run.get('steps')!=300", text)
        self.assertIn("association=associate(owner,report,rgb,ids,detections,mapped)", text)
        self.assertIn("association['accepted']==2", text)

    def test_preflight_never_reads_held_out_pixels_or_imports_torch(self):
        text = (c.HERE / "mask_resolution_campaign.py").read_text()
        self.assertIn("deny_held_out(manifest[\"records\"])", text)
        self.assertIn("if torch.cuda.is_available()", text)
        self.assertIn("MAX_WALL = 10800", text)
        self.assertIn("MAX_RSS_MIB = 6144", text)
        self.assertIn("memory_mib=MAX_RSS_MIB", text)
        self.assertIn('return 0 if result["status"] in ("VALIDATION_PASS", "TRAINED_REJECTED")', text)


if __name__ == "__main__":
    unittest.main()
