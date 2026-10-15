# Native boundary candidate 001 — rejected

The separately authorised single campaign completed 300/300 updates and the
preregistered full 12-scene validation, using unchanged commit
`6cbb12b86c4351676eb4822d674f90ba5ac5b33e`. **Candidate FAIL / REJECTED.**
No retry, parameter change, additional epoch or additional experiment occurred.
This is a training-only development result; Stage-A2 remains **BLOCKED 0/2**
and original 56-grid training supervision remains **UNKNOWN**.

## Admission and genuine supervision

The clean PR229 branch was fetched and fast-forward checked (already current).
The exclusive output did not exist before launch. All29 source/runtime bindings,
9 evidence bindings, initializer, dataset, runtime inventory and12 baseline
archives passed exact checks. Configuration SHA256:
`1b753bf1bca3a6466521459375c4d8f3e4cf900a6a905fc8750520a431263e6f`.
Independent implementation review SHA256:
`fcb9eb79b5f0255d95a006685b71bcb70b0a5f78d41aaa9d80474b6cd6f634c6`.
The exact previously documented command was invoked once with explicit human
authorisation. No tests were rerun: the previously reviewed97/97 PASS result
remains historical implementation evidence, not a new campaign measurement.

Recipe: seed260110,300 updates,batch1,SGD LR0.0001/momentum0.9/decay0.0001,
ROI mask loss2,auxiliary0.25,native boundary radius2px. Initializer was the
SHA-bound #221epoch02 checkpoint adapted to56-grid, not the provenance-unknown
56-grid baseline. CPU-only,two threads; caps10800s/6144MiB worker RSS and5GiB
disk reserve. The live RUNNING record preceded training. Independent reviewer
observed parent44898/worker44906, separate PID=PGID=SID worker session, exact
launch binding and CUDA-disabled/two-thread environment.

Observed worker OS exit **0**, supervisor CLI exit **2** (candidate rejection,
not training failure). Wall **2161.0899834059965s** including validation, peak
**3676856KiB (3.506523GiB)**, no resource-limit reason, no supervision failure,
process-group cleanup true. RSS enforcement was20ms VmRSS/VmHWM polling plus
wait4 high-water, not a kernel cgroup hard cap. Durable logs, RUNNING receipt,
completed supervision and parent completion receipt are retained. Worker result
correctly retains AWAITING_PARENT_RECEIPT because it predates parent completion;
completion.json provides the later genuine receipt. No historical receipt was
created or changed.

History has300 finite sequential updates, the frozen seeded scene order and
checkpoint/history/validation/source bindings. Frozen non-mask tensors and
buffers remain byte-identical before/after:
`ffa1d1d9dda33d728ad8e13c30433d51d5d5ad305ebd2b8aa2c113c39be1d72f`.
No held-out or Stage-A2 scene access; only approved training and validation.

## Paired validation

Metrics use the preregistered consumer ROI, strict unchanged threshold and
original conservative ANY2x2 mapping, with baseline-frozen assignments. They
must not be conflated with historical full-image mask metrics.

| Measure | Frozen baseline | Candidate | Decision |
|---|---:|---:|---|
| Native512 background spill |1193|888|25.5658% reduction; materiality PASS|
| Mapped256 background spill |647|527|18.5471% reduction; materiality FAIL|
| Native sentinel spill,024–027 |441|328|cap352 PASS|
| Mapped sentinel spill,024–027 |214|175|cap171 FAIL|
| Native missing foreground |5620|5846|regression FAIL|
| Mapped missing foreground |1183|1207|regression FAIL|
| Native cross-object pixels |1711|1628|no regression|
| Mapped cross-object pixels |537|496|no regression|
| Native mean instance IoU |0.738880883493|0.740774663923|aggregate improves; individual failures|
| Mapped mean instance IoU |0.726147466879|0.733929184688|aggregate improves; individual failure|
| Recovered instances |18/22|18/22|unchanged|
| Confident detections on negative scenes024/030 |0/0|0/0|PASS|

All scene background-spill totals decreased or stayed zero. Native missing
foreground increased for17 instances; mapped missing increased for9. Native
IoU decreased for6 instances and mapped IoU for1. No instance cross-object
regression. Aggregate improvements do not override any per-instance or
per-detection regression. For example,scene025/65508 native missing50→78,
mapped4→9;scene032/65496 native IoU0.839863713799→0.834754595414,
mapped0.863465866467→0.861654135338. Frozen boxes/labels/scores were bitwise
identical across all12 paired prediction archives.

| Scene | Native baseline→candidate spill | Mapped baseline→candidate spill |
|---|---:|---:|
|024|0→0|0→0|
|025|91→53|45→32|
|026|262→224|119→104|
|027|88→51|50→39|
|028|150→109|85→68|
|029|117→92|58→48|
|030|0→0|0→0|
|031|136→96|68→55|
|032|153→125|82→68|
|033|30→18|35→27|
|034|49→34|36→25|
|035|117→86|69→61|

Exact gate failures: DETECTION_REGRESSION, INSTANCE_REGRESSION,
MATERIALITY_mapped, SENTINEL_MATERIALITY_mapped. The candidate is rejected,
regardless of successful training supervision. This single experiment supports
an FP/FN tradeoff at this fixed loss/configuration; it does not isolate which
loss component or initializer difference caused it, nor justify tuning against
acceptance data. No production qualification or deployment is implied.

## Evidence and independent verification

External immutable run: `/home/user/epd-v2-artifacts/native-boundary-candidate-001`.
Repository receipt/metrics/history/log copies and every external file hash:
`evidence/segmentation_v2/native_boundary_candidate_001/artifact_manifest.json`.
Checkpoint `UNQUALIFIED_epoch_01.pt` SHA256:
`38bc0d71a5f11cbf5ef567403f4a18c8aeb1d9361044a05c48cd93eb8cdd73b3`.
History SHA256:
`a55a5fa96f6431fa5d82b783079e4ef419a65f5864672c564ed7de50aa8381f3`.
Validation SHA256:
`4a40df1a135c09620e92038062daed39913be2b447f8e89f45ad669b98066f71`.

Independent post-run verification is recorded in the accompanying
`independent_review.json`; it reproduces the scalar acceptance decision without
training or inference. Historical artifacts and source/configuration are unchanged.

**Single next action:** accept the candidate rejection and retain the frozen
baseline; no further campaign is authorised.
