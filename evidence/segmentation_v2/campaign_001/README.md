# Campaign 001 — V2_TRAINED_BUT_REJECTED

One complete deterministic CPU campaign ran eight epochs / 192 optimizer steps. All eight checkpoints failed the frozen validation gate. No checkpoint was accepted, exported, or consumed through ONNX/C++; no parity result is claimed. PR #219 remains draft.

## Actual execution

- Starting source: `0e83096cd239c352e5cd384eb5c46e66945da5fd`. Frozen policy: `tools/segmentation_v2/campaign.json`, SHA256 `1820832149bfc4931c0ee36e800b9f9d2273bdc6c49355408c718b44a7756360`.
- CUDA unavailable (zero devices, no NVIDIA device nodes); existing torch 2.7.0+cu126 / torchvision 0.22.0. No dependencies installed or upgraded.
- CPU budget declared before measurement: 3600 s / 6144 MiB RSS, 5 GiB disk reserve. Preflight estimated 852.810 s and 2054148 KiB peak (1.5 safety factor); actual process 579.083 s, sampled peak 2059920 KiB (1.965 GiB), exit 0. Initial free disk 87.265 GiB. CUDA policy remains 1800 s.
- Fresh hash-pinned COCO_V1 initialization; frozen ResNet50-FPN backbone, newly initialized two-class box/mask heads, SGD lr 0.001 / momentum 0.9 / weight decay 0.0001, batch 1, seed 260110, two CPU threads, deterministic PyTorch. The rejected two-step checkpoint was hash-checked only and never loaded.
- Verified 360 train/validation acquisition files, 24/12/12 frozen membership and all 48 successful capture outcomes from manifests. Opened only 24 train and 12 validation scene files. Native mappings, classes and visual/link/model ownership independently audited. Acquisition SHA matches the committed pilot manifest.
- No new acquisition, augmentation, threshold changes, renderer-ID model inputs or runtime mask correction. Native IDs serve only as supervised targets and evaluation references. RGB remains the existing bilinear 256→512 RGB float32 /255 contract, with synchronous native512 instance labels.

## Frozen validation decision

Gate fixed before training: recovered ≥18/22, mean IoU including misses >0.685890, false positives ≤1, background spill <1915, zero scoped class/identity/output-contract violations. Confidence ≥0.8; masks strictly >0.5. Recovery requires one-to-one IoU ≥0.5. Eligible ordering: recall count, mean IoU, lower spill, earlier epoch. No eligible checkpoint exists.

| Checkpoint | Recall | Mean IoU incl. misses | FP | Spill px | Cross-object px | Missing px | Mean boundary px* | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Historical baseline | 17/22 | 0.685890 | 1 | 1915 | 914 | 8239 | 117.543601 | reference |
| Epoch 1 | 0/22 | 0.000000000 | 0 | 0 | 0 | 58237 | 512.000000 | no |
| Epoch 2 | 0/22 | 0.000000000 | 0 | 0 | 0 | 58237 | 512.000000 | no |
| Epoch 3 | 0/22 | 0.000000000 | 0 | 0 | 0 | 58237 | 512.000000 | no |
| Epoch 4 | 1/22 | 0.037938357 | 0 | 77 | 0 | 54628 | 488.927883 | no |
| Epoch 5 | 4/22 | 0.165669791 | 0 | 380 | 188 | 44063 | 419.131814 | no |
| Epoch 6 | 5/22 | 0.217194139 | 1 | 339 | 147 | 41941 | 373.255515 | no |
| Epoch 7 | 6/22 | 0.248926434 | 0 | 266 | 249 | 41241 | 372.712662 | no |
| Epoch 8 | 6/22 | 0.250119941 | 0 | 238 | 196 | 39698 | 372.713272 | no |

*Boundary is symmetric contour distance; missed instances receive the pre-existing 512 px penalty. Per-instance measured distances and null distances for misses are retained in `offline_audit.json`. Background/cross-object counts sum all confident detection masks; missing counts use one-to-one assigned instances. All epochs fail recall and IoU; none failed the FP or spill limits. Zero scoped violations means valid output tensors/classes and audited native ownership, not zero cross-object contamination or successful EMD association.

Both validation negative images (`scene_024`, `scene_030`) had zero confident detections at every epoch. All raw outputs, including below-confidence detections, are hash-pinned in `artifact_locations.json`. `validation_XX.json` preserves per-instance/detection metrics and timing. `offline_audit.json` independently verifies checkpoint/raw-output hashes, every epoch exactly once, contiguous steps and every training scene exactly once per epoch, and reproduces the recorded gate metrics exactly.

## Provenance and failures

- The worker executed the exact `executed_campaign.py` snapshot (SHA recorded in `run_config.json`), not the subsequently hardened current source. Review added source-hash enforcement and explicit identity auditing to future launches; independent offline checks verify those facts for this completed run. Offline auditing also preserves exact zero boundary errors instead of treating them as missing. It changes no gate metric, prediction or eligibility decision.
- An initial strict comparison of prior smoke/config identities failed: its historical `validation_selection` description predates the corrected FP-aware diagnostic description. `source_identity.json` records the sole difference and exact byte reconstruction of the historical SHA. Current recipe bytes match the requested starting commit; campaign selection uses the separate frozen user gate.
- Checkpoint selection rejected every epoch. Epoch 8 is the strongest diagnostic checkpoint only: recall 27.27%, mean IoU 0.2501199405210875, spill 238, cross-object 196, missing visible 39698, zero FP. It is not a qualified initialization or export candidate.
- Starting-head CI has three failures, each with identical root errors on base main `432a15317ff899fa8636533bf24dc44369b360b0`: CLI inherited W503 lint; GUI unsupported pycodestyle `--extend-ignore`; ROS missing NumPy in rosidl_generator_py/epd_msgs. `ci_results.json`, `ci_comparison.json` and six compressed actual job logs retain comparisons. No introduced regression identified, workflow refactoring or green CI claim.

## Retained artifacts and verification

Large artifacts remain at `/home/user/epd-v2-artifacts/campaign-001/training` on this host, outside Git. This is retained local storage, not a claim of remote artifact hosting. All eight model+optimizer/RNG checkpoints and 96 raw validation NPZs have paths, sizes where available, and full SHA256 references in `artifact_locations.json`. Loss curves/history, source/config provenance and process output are committed here. `executed_preflight.py` is the exact host-specific measurement script; `preflight_process.json` records its bounded invocation.

Epoch 8 SHA256: `b284ee6e64b3f75a4cc9a2bac0ec997916827cb4a982d5ad0c840e80ac9bda8a`. ONNX SHA: null (not exported after rejection). The existing adapter/export contract in `docs/segmentation_v2_adapter.md` remains intact; PyTorch→ONNX→C++ verification must await an eligible candidate. Authoritative EMD association counts remain unexecuted.

Focused tests: 14 passed, including frozen gate, output validation, negatives, thresholds, identity ownership, exact offline replay and rejection of incomplete checkpoint/history records or tampered hashes. Fresh command output is in `verification.txt`.

## One next action

Acquire the remaining 276 genuine training scenes toward the documented 300 training / 100 validation / 100 held-out scale in a separately authorized campaign. Current counts remain 24/12/12. Do not reuse this campaign’s rejection as evidence of production qualification. Frozen held-out images and the historical two-cube input were not opened/evaluated. EMD, conservative geometry, fail-closed association and the 0.1 mm physical-contact requirement remain unchanged. No motion, controllers, MoveIt, merge or deployment occurred.
