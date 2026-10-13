#!/usr/bin/env python3
"""Exactly one supervised CPU epoch on 56-grid masks. No export or promotion.

This intentionally does not call the 4-update disposable feasibility worker.
Fail closed on artifact/version/resource drift and preserve partial evidence.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time

import numpy as np

from pilot import HERE, sha, write_json, bounded
from mask_resolution import INITIAL_SHA, DATASET_SHA, configure56
from original_finetune import deny_held_out
from recover_original import safe_load, reconstruct
from train import tensors
from campaign import campaign_rows, epoch_order, validate_epoch
from continuation import atomic_checkpoint


EVIDENCE = HERE.parents[1] / "evidence/segmentation_v2/mask_resolution_001"
MAX_WALL = 10800
MAX_RSS_MIB = 6144
MIN_FREE_BYTES = 6 * (1024 ** 3)  # 5GiB reserve plus up to 1GiB outputs
SEED = 260110

POLICY = {
    "schema": "epd_56_grid_one_epoch/v1",
    "seed": SEED,
    "epochs": 1,
    "training_samples": 300,
    "validation_samples": 12,
    "held_out_samples": 12,
    "batch_size": 1,
    "cpu_threads": 2,
    "workers": 0,
    "learning_rate": 0.0001,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "mask_loss_weight": 2.0,
    "mask_roi_output_size": 28,
    "mask_logits_size": 56,
    "frozen_non_mask_parameters": True,
    "gate": {
        "positives": 22,
        "minimum_recovered": 18,
        "mean_iou_strict_gt": 0.685890,
        "maximum_false_positives": 1,
        "background_spill_strict_lt": 1915,
        "maximum_contract_violations": 0,
        "confidence_threshold": 0.8,
        "mask_threshold_strict_gt": 0.5,
    },
    "reference_epoch2": {
        "recovered": 18, "positives": 22, "mean_iou": 0.758658574,
        "false_positives": 1, "background_spill": 1185,
        "stage_a2_accepted": 0,
    },
    "checkpoint_sha256": INITIAL_SHA,
    "dataset_sha256": DATASET_SHA,
    "maximum_wall_seconds": MAX_WALL,
    "maximum_rss_mib": MAX_RSS_MIB,
    "disk_reserve_gib": 5,
    "held_out_access": False,
    "stage_a2_access_during_selection": False,
    "automatic_retry": False,
}


def evidence_admission(*, checkpoint: Path, artifacts: Path, output: Path) -> dict:
    """Revalidate the *original* recorded CPU evidence, not a new benchmark."""
    if output.exists():
        raise ValueError("OUTPUT_NOT_EXCLUSIVE: use a new nonexistent campaign directory")
    checkpoint = checkpoint.resolve(strict=True)
    artifacts = artifacts.resolve(strict=True)
    if sha(checkpoint) != INITIAL_SHA or sha(artifacts) != DATASET_SHA:
        raise ValueError("TRAINING_INPUT_DRIFT: #221 checkpoint or frozen dataset hash differs")
    source = json.loads((EVIDENCE / "unattempted_result.json").read_text())
    finished = json.loads((EVIDENCE / "unattempted_process.json").read_text())
    failed = json.loads((EVIDENCE / "failed_process.json").read_text())
    if (source.get("status") != "FEASIBILITY_PASS" or
            source.get("source_sha256") != sha(HERE / "mask_resolution.py") or
            source.get("initialization_sha256") != INITIAL_SHA or
            source.get("dataset_sha256") != DATASET_SHA or
            source.get("benchmark_updates") != 4 or
            source.get("training_epochs") != 0 or
            source.get("campaign_updates") != 0 or
            source.get("accuracy_evaluated") is not False or
            source.get("qualified_model") is not False or
            source.get("stage_a2_inference") is not False or
            finished.get("exit_code") != 0 or finished.get("resource_limit_reason") is not None or
            failed.get("exit_code") != 1 or failed.get("resource_limit_reason") is not None):
        raise ValueError("FEASIBILITY_EVIDENCE_INVALID: original pinned probe no longer qualifies")
    step = source.get("step_time_upper_bound")
    if (not isinstance(step, (int, float)) or isinstance(step, bool)
            or not math.isfinite(step) or step < failed["wall_seconds"] - 1e-6
            or step < max(row["seconds"] for row in source["measurements"])):
        raise ValueError("FEASIBILITY_EVIDENCE_INVALID: unbounded measured step time")
    estimated = (300 + 24) * step * 1.5 + 240
    if (not math.isclose(estimated, source["projected_one_epoch_seconds"], abs_tol=1e-4)
            or estimated > MAX_WALL
            or source["projected_two_epoch_seconds"] <= MAX_WALL):
        raise ValueError("RESOURCE_ADMISSION_FAILED: one-epoch projection invalid or over budget")
    peak = max(failed["child_peak_rss_kib"], finished["child_peak_rss_kib"])
    estimated_rss_kib = 1.5 * peak
    if (not math.isclose(estimated_rss_kib, source["projected_rss_kib"], abs_tol=1)
            or estimated_rss_kib > MAX_RSS_MIB * 1024):
        raise ValueError("RESOURCE_ADMISSION_FAILED: measured RSS drift/over budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    if free < MIN_FREE_BYTES:
        raise ValueError("RESOURCE_ADMISSION_FAILED: require 5GiB reserve and 1GiB write allowance")
    if os.cpu_count() is None or os.cpu_count() < 2:
        raise ValueError("RESOURCE_ADMISSION_FAILED: CPU worker requires at least two cores")
    # This is not a second execution of the expensive shape test. The model
    # construction and actual tensor-level shape check occur inside worker.
    return {
        "status": "GO_ONE_EPOCH_ONLY",
        "source_evidence_sha256": {name: sha(EVIDENCE / name) for name in (
            "unattempted_result.json", "unattempted_process.json", "failed_process.json")},
        "checkpoint_sha256": INITIAL_SHA, "dataset_sha256": DATASET_SHA,
        "projected_seconds": estimated, "projected_rss_kib": estimated_rss_kib,
        "wall_limit_seconds": MAX_WALL, "memory_limit_mib": MAX_RSS_MIB,
        "available_disk_gib": free / 1024 ** 3,
        "training_updates_authorized": 300, "training_epochs_authorized": 1,
        "validation_scenes": 12, "stage_a2_inference": False,
        "held_out_access": False, "model_qualified": False,
    }


def run_worker(args) -> None:
    import torch

    # Protect against starting the worker manually with altered/deleted inputs,
    # a source modification after parent preflight, or an incompatible runtime.
    admission_path = args.output / "admission.json"
    admission = json.loads(admission_path.read_text())
    for name, digest in admission["source_evidence_sha256"].items():
        if sha(EVIDENCE / name) != digest:
            raise ValueError("FEASIBILITY_SOURCE_CHANGED: retained evidence changed after admission")
    binding = json.loads((args.output / "binding.json").read_text())
    for name, digest in binding["sources"].items():
        if sha(HERE / name) != digest:
            raise ValueError("TRAINING_SOURCE_CHANGED: source file changed after admission")
    if sha(args.checkpoint) != INITIAL_SHA or sha(args.artifacts) != DATASET_SHA:
        raise ValueError("TRAINING_INPUT_DRIFT: input changed after preflight")
    for name, expected in (("torch", "2.7.0"), ("torchvision", "0.22.0"),
                           ("numpy", "1.26.4"), ("opencv-python", "4.11.0.86")):
        if importlib.metadata.version(name).split("+")[0] != expected:
            raise ValueError("DEPENDENCY_DRIFT: " + name)
    if torch.cuda.is_available():
        raise ValueError("CPU_ONLY: GPU detected; not this admitted experiment")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    manifest = json.loads(args.artifacts.read_text())
    deny_held_out(manifest["records"])
    train, val = campaign_rows(args.artifacts, POLICY)
    if len(train) != 300 or len(val) != 12:
        raise ValueError("FROZEN_SPLIT_INVALID: expected 300 train and 12 validation")
    original = safe_load(args.checkpoint, INITIAL_SHA)
    if not isinstance(original, dict) or "model" not in original:
        raise ValueError("CHECKPOINT_INVALID: #221 state_dict unavailable")
    original_state = original["model"]
    model = configure56(reconstruct(original_state))
    if len(model.state_dict()) != 307:
        raise ValueError("ARCHITECTURE_DRIFT: expected 307 checkpoint tensors")
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if len(trainable) != 12 or not all(name.startswith(
            ("roi_heads.mask_head.", "roi_heads.mask_predictor.")) for name in trainable):
        raise ValueError("PARAMETER_FREEZE_FAILED: only twelve mask tensors may train")

    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                                lr=POLICY["learning_rate"], momentum=POLICY["momentum"],
                                weight_decay=POLICY["weight_decay"])
    write_json(args.output / "configuration.json", {
        "policy": POLICY, "architecture": "mask_roi28_logits56",
        "trainable_names": trainable, "fresh_optimizer": True,
        "source_binding": binding, "dataset_sha256": DATASET_SHA,
        "initial_checkpoint_sha256": INITIAL_SHA, "held_out_files_opened": 0,
    })

    # Frozen actual order and a new exclusive history. A killed/interrupted run
    # cannot be resumed as an ineligible partial epoch.
    start = time.monotonic()
    history_path = args.output / "history.jsonl"
    history_path.touch(exist_ok=False)
    history = []
    model.train()
    for count, index in enumerate(epoch_order(len(train), SEED, 1), start=1):
        if shutil.disk_usage(args.output).free < MIN_FREE_BYTES:
            raise ValueError("RESOURCE_DISK_RESERVE: stop before falling below retained reserve")
        row, folder, report, ids = train[index]
        image, target = tensors(folder, report, ids, torch)
        optimizer.zero_grad(set_to_none=True)
        then = time.monotonic()
        losses = model([image], [target])
        if "loss_mask" not in losses:
            raise ValueError("MASK_LOSS_MISSING: cannot train without actual supervised masks")
        total = sum(value * (2 if key == "loss_mask" else 1)
                    for key, value in losses.items())
        if not torch.isfinite(total):
            raise ValueError("TRAINING_LOSS_NONFINITE: preserve incomplete evidence")
        total.backward()
        optimizer.step()
        entry = {
            "step": count, "epoch": 1, "scene_id": row["scene_id"],
            "seconds": time.monotonic() - then,
            "loss": float(total.detach()),
            **{key: float(value.detach()) for key, value in losses.items()},
        }
        with history_path.open("a") as out:
            out.write(json.dumps(entry, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())
        history.append(entry)
        del total, losses, target, image, ids
        if count % 50 == 0:
            print(json.dumps({"completed_updates": count, "expected_updates": 300,
                              "elapsed_seconds": time.monotonic() - start}), flush=True)
    if len(history) != 300 or len(set(row["scene_id"] for row in history)) != 300:
        raise ValueError("TRAINING_HISTORY_INCOMPLETE: missing or duplicated update")

    # Verify non-mask parameters and buffers were never changed, including BN
    # state, before publishing any candidate checkpoint.
    unchanged = all(torch.equal(before, model.state_dict()[name])
                    for name, before in original_state.items()
                    if not name.startswith(("roi_heads.mask_head.", "roi_heads.mask_predictor.")))
    if not unchanged:
        raise ValueError("NON_MASK_STATE_CHANGED: trained model violates frozen architecture")
    validation = validate_epoch(model, val, "cpu", torch,
                                args.output / "validation_01", 1, POLICY)
    np_state = np.random.get_state()
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": 1, "step": 300, "seed": SEED,
        "architecture": "mask_roi28_logits56",
        "mask_roi_output_size": 28, "mask_logits_size": 56,
        "policy": POLICY, "binding": binding,
        "history_sha256": sha(history_path),
        "validation_sha256": sha(args.output / "validation_01/metrics.json"),
        "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
        "numpy_rng_state": dict(algorithm=np_state[0], keys=np_state[1].tolist(),
                                 pos=np_state[2], has_gauss=np_state[3], cached_gaussian=np_state[4]),
    }
    checkpoint = atomic_checkpoint(args.output / "epoch_01.pt", state, torch)
    outcome = "VALIDATION_PASS" if validation["eligible"] else "TRAINED_REJECTED"
    result = {
        "schema": "epd_56_grid_one_epoch/v1", "status": outcome,
        "architecture": "mask_roi28_logits56",
        "steps": 300, "epochs": 1, "training_validation_seconds": time.monotonic() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "selected": checkpoint if validation["eligible"] else None,
        "checkpoints": [checkpoint], "epoch_metrics": [validation],
        "reference_epoch2": POLICY["reference_epoch2"],
        "held_out_files_opened": 0, "stage_a2_used_for_selection": False,
        "stage_a2_accepted": None, "exported": False, "production_qualified": False,
    }
    write_json(args.output / "result.json", result)
    print(json.dumps({key: result[key] for key in (
        "status", "steps", "epochs", "selected", "training_validation_seconds")}), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "worker"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "/home/user/epd-v2-artifacts/mask-finetune-001/epoch_02.pt"))
    parser.add_argument("--artifacts", type=Path, default=Path(
        "/home/user/epd-v2-artifacts/training-expansion-002/integration/dataset.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "worker":
        run_worker(args)
        return 0
    try:
        admitted = evidence_admission(checkpoint=args.checkpoint, artifacts=args.artifacts, output=args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), flush=True)
        return 2
    if args.action == "check":
        print(json.dumps(admitted, sort_keys=True), flush=True)
        return 0
    args.output.mkdir(exist_ok=False)
    write_json(args.output / "admission.json", admitted)
    names = ("mask_resolution_campaign.py", "mask_resolution.py",
             "mask_finetune.py", "recover_original.py", "original_finetune.py",
             "pilot.py", "train.py", "campaign.py", "continuation.py",
             "dataset.py", "dataset_bindings.py", "expansion.py")
    source_binding = {name: sha(HERE / name) for name in names}
    snapshot = args.output / "executed_sources"
    snapshot.mkdir(exist_ok=False)
    for name in names:
        shutil.copy2(HERE / name, snapshot / name)
    write_json(args.output / "binding.json", {
        "sources": source_binding, "checkpoint": INITIAL_SHA,
        "dataset": DATASET_SHA, "policy": POLICY,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "torchvision", "numpy", "opencv-python")},
    })
    command = [sys.executable, str(Path(__file__).resolve()), "worker",
               "--checkpoint", str(args.checkpoint.resolve()),
               "--artifacts", str(args.artifacts.resolve()),
               "--output", str(args.output.resolve())]
    outcome = bounded(command, args.output, MAX_WALL,
                      dict(os.environ, OMP_NUM_THREADS="2"), memory_mib=MAX_RSS_MIB)
    write_json(args.output / "process.json", outcome)
    print(json.dumps(outcome, sort_keys=True), flush=True)
    if outcome["exit_code"] != 0 or not (args.output / "result.json").is_file():
        return 2
    result = json.loads((args.output / "result.json").read_text())
    if result.get("steps") != 300 or result.get("epochs") != 1:
        return 2
    return 0 if result["status"] in ("VALIDATION_PASS", "TRAINED_REJECTED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
