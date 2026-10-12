# Original model fine-tuning resource admission

**BLOCKED: projected campaign wall time exceeds the declared budget. No campaign started.**

Branch `codex/epd-original-model-finetune` starts at recovery commit `d841b522e797aebe55f0bc1ff288f970cffaad85`. Its draft PR is stacked on `codex/epd-v2-supervised-pilot` (#219); #219 is unchanged. No architecture investigation, training campaign, export, held-out evaluation or deployment was repeated.

## Verified inputs and configuration

- Dataset manifest `/home/user/epd-v2-artifacts/training-expansion-002/integration/dataset.json`, SHA256 `04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11`. Exactly **300 training scenes, 484 visible cube instances and 50 negatives**; 12 validation scenes with 22 visible instances; 12 test memberships verified without opening test files. All training/validation artifact hashes and native label ownership mappings passed. All 324 IDs/seeds and both RGB hash inventories are unique, with no split overlap or historical regression RGB hashes. Training tensors/masks passed the existing integrity checks. No data acquisition, regeneration or label changes.
- Original checkpoint `cube_maskrcnn_checkpoint_thresh_08.pth`, SHA256 `b2668971dd1230cdbc93bbc7d0182b46c91e7469717dc210052fa4eb20fa8bdb`, unchanged. CPU `weights_only=True` loading, all 307 state tensors strict-loaded. Original two-class heads retained: 0 background, 1 cube. Recovered FrozenBatchNorm epsilon 0 and RGB256 → INTER_LINEAR RGB512 → CHW float32 /255 retained; original model owns mean/std normalization and internal min800/max1333 resize. No COCO or rejected replacement weights loaded.
- Original ONNX SHA256 `21362d62d816bd684f2b5c7769d0bcd1cd568af86419788fca9d095c46f2bad2` verified unchanged. This task does not repeat the recovery parity test.
- Preregistered `tools/segmentation_v2/original_finetune.json`: eight epochs, 300 scenes/epoch, 2,400 updates; batch 1, workers 0, float32, two CPU threads, seed 260110; SGD lr0.001, momentum0.9, weight decay0.0001, no scheduler, frozen backbone. These optimization settings retain the existing recipe; initialization and recovered original transform differ explicitly. The original historical optimizer/freeze policy remains unknown. Any eventual campaign must start freshly from the original checkpoint, not the disposable benchmark state.
- Finite admission policy: 18,000s full training/evaluation wall limit, 6GiB RSS, 5GiB reserve plus 8×400MiB checkpoint allowance. Projection uses maximum measured complete SGD step ×2,400 plus maximum validation forward ×108 (original baseline + eight epochs), multiplied by the existing 1.5 safety factor, plus 240s startup/checkpoint/scoring overhead. No change to the prior failed policies or evidence. The campaign runner/continuation adaptation is deferred because admission failed; this branch provides only the bounded admission path.

## Actual measurements and all failures

Hardware: Intel Core i7-8750H guest, eight virtual CPU cores; CUDA unavailable, Torch CUDA build 12.6, zero CUDA devices. No NVIDIA tooling. Approximately 11.2GiB available RAM before probing; the benchmark recorded 10.220GiB available after dataset/model preparation and 82.207GiB free disk. Installed Torch2.7.0, torchvision0.22.0, NumPy1.26.4, ONNX1.17.0, ORT1.22.0; exact identities in evidence. No dependency installation or upgrade.

A disposable original-weight model benchmark used a negative, two largest three-object scenes and the smallest visible-object scene. Selection used training geometry/label area, never predictions or held-out inputs. Each measured step includes forward, all segmentation losses, backward and SGD update. Two validation forwards measure runtime only; their predictions are discarded and are not quality evidence. Images alone enter inference; native IDs remain target/scoring data.

| Measurement | Actual |
|---|---:|
| Four complete SGD steps |3.246794, 4.888313, 4.042337, 4.877977s|
| Two validation forwards |2.714741, 2.864536s|
| Native successful-measurement process wall |40.812932s|
| Child peak RSS |1,951,144KiB (1.861GiB)|
| Monitor sampled peak RSS |1,951,832KiB|
| Projected RSS with safety factor |2,926,716KiB (2.791GiB)|
| Required free disk including reserve |8,724,152,320 bytes|
| Full-campaign projected wall |**18,301.979934s**|
| Declared limit |18,000s|
| Admission |**BLOCKED: wall_time**|
| Campaign epochs / updates / duration |**0 / 0 / 0s**|

Probe001 stopped after its second disposable SGD update: `pilot.write_json` uses exclusive creation and the new measurement logger reused `measurements.json`. Only the first timing was published. Native exit1, 32.561892s, child peak1,886,916KiB; no resource-limit termination. Fixed by exclusive per-step files, with a failing-then-passing regression test. Its original outputs and executed-source snapshots remain intact. Probe002 completed all four disposable updates and two timed forwards, exited2 intentionally for failed admission, with no resource-limit termination. No weights or optimizer states were saved from either probe; six disposable benchmark updates in total, zero campaign updates. No retry after resource rejection, shorter campaign, safety-factor reduction or alternate hyperparameter attempt.

## Quality and compatibility status

| Metric | Retained historical original baseline | New candidate |
|---|---:|---|
| Recall at confidence ≥0.8 |17/22 (77.27%)|Not measured; no candidate|
| Mean instance IoU including misses |0.685890|Not measured|
| False positives |1|Not measured|
| Background spill |1,915 pixels|Not measured|
| Box/mask AP, precision, per-class/occlusion measurements |No new baseline evaluation in this task|Not measured|

The baseline numbers are the frozen historical validation reference, **not** measurements from the disposable benchmark. No baseline-versus-candidate inference comparison or quality selection occurred. Frozen gates remain recall≥18/22, mean IoU>0.685890, FP≤1, spill<1,915, zero class/identity/output violations, confidence≥0.8 and mask foreground strictly>0.5. Existing ranking is unchanged. No candidate is accepted or exported. Original recovery's one-input parity remains the only newly recovered-model parity evidence; no new ONNX or EPD C++ compatibility claim.

## Reproduction and next milestone

Run only into a new exclusive external directory, never overwrite either retained probe:

```bash
python3 tools/segmentation_v2/original_finetune.py measure \
  --checkpoint /home/user/epd-v2-artifacts/original-cube-checkpoint/cube_maskrcnn_checkpoint_thresh_08.pth \
  --artifacts /home/user/epd-v2-artifacts/training-expansion-002/integration/dataset.json \
  --output /home/user/epd-v2-artifacts/<new-exclusive-admission-directory>
python3 -m unittest discover -s tools/segmentation_v2 -p 'test_*.py'
```

The probe is independently bounded at240s/6GiB and rejects held-out file access using a Python audit hook. **47 focused tests pass**, including conservative projection, nonfinite measurements, frozen gates and exclusive measurement logging. Existing ROS `launch.actions` and GPU-only repository collection blockers remain separate; repository-wide pytest was not rerun because this branch only changes the standalone training tooling. Prior recovery evidence retains those collection failures.

Independent read-only review reproduced the admission arithmetic and verified all42 external hashes, with no critical/important findings. Its minor custom-policy snapshot-path issue was fixed after measurement: future probes copy the actual supplied policy path. The exact earlier executed sources remain hash-pinned in each external snapshot; the fixed source was not used to repeat the rejected resource probe.

Large artifacts remain outside Git. `evidence/segmentation_v2/original_finetune_001/artifact_locations.json` records sizes/SHA256s for both probes, exact executed sources, original checkpoint/ONNX and dataset manifest. Training/validation file hashes and acquisition manifests remain bound through the unchanged dataset; test pixels were never opened.

**Next milestone:** obtain resource admission for this original-weight eight-epoch campaign before implementing/running its hash-bound checkpoint-resumable continuation. At the measured validation time, the same18,000s policy requires maximum SGD-step time≤4.804429s (observed4.888313s), or an explicitly reviewed finite wall budget of at least18,302s on this CPU. No budget change is applied here. An admitted run must then evaluate the unchanged original baseline and candidates with identical preprocessing, add box/mask AP where meaningful, enforce the frozen gate, and conditionally run ONNX/isolated EPD compatibility verification.

No robot motion, Workcell Studio deployment, EMD modifications or changes to conservative collision/contact qualification occurred. Both PRs remain unmerged; #219 remains draft and untouched.
