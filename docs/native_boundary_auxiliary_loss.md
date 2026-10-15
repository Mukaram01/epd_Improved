# Native512 boundary auxiliary loss — implementation, not training

## Outcome and scope

This PR stacks on verified#228 commit`e7786212f23c384b0ff00c422c472e7f261e134b`.
It implements one training-only candidate: native512 post-paste boundary FP/FN
supervision. Original inference architecture, transforms, boxes, scores,
confidence>=.8, strict mask>.5, paste geometry, ANY2×2 and EMD are unchanged.
All historical scripts and experiment directories remain untouched.

**No campaign, optimizer update, trained checkpoint load, real-scene inference,
held-out evaluation or Stage-A2 access occurred.** Tests use invented binary
annotations and an artificial4-channel network/64internal raster, exercising
installed Torchvision RoIHeads, ROI28→56 and native512 paste. The independent
gradient check uses backward only; all toy state remains unchanged.
Stage-A2 remains **BLOCKED0/2**; original56training supervision **UNKNOWN**.

## Loss and integration contract

`native_boundary_aux.training_losses(model,images,targets,weight,radius)` is
called only by the new prospective runner. **Weight0 returns the original
model forward directly**, with no target cloning, hooks, native loss or new key.
At positive weight, original stock ROI56 mask loss and all other stock losses
are still computed by installed RoIHeads unchanged. The only added dictionary
entry is`loss_mask_native_boundary = weight * L_native`.

Temporary **per-model** hooks capture transformed targets/image sizes, the
actual positive mask proposals, and differentiable logits. No process-global
maskrcnn_loss monkeypatch or inference replacement is installed. Hooks and all
module training flags are restored on success/failure. Calls are CPU-only,
single Python forward at a time; this is not a concurrent/DDP integration.

Native truth is cloned **before** image transformation. Transformed labels,
boxes and nearest-resized masks must equal the original instance ordering;
reordered/substituted ownership fails closed. The actual per-model matcher is
reused on the exact captured positive proposals, with low-quality matching
explicitly forbidden. The installed default matcher is independent per proposal,
including first-GT tie resolution. Empty positive lists skip rematching.
Only each proposal's matched class/instance is supervised, never a target union.
Visible native masks must be binary512×512, mutually exclusive, have positive
class labels and valid containing boxes. Invisible objects are omitted; empty
negative-image targets are supported. Fragmented/partially occluded masks and
single-pixel-thin features are valid.

For each sampled proposal:

1. Resize its detached **transformed** box back to native512 with original
   `resize_boxes`, not the GT box or a decoded inference box.
2. Select its matched class's actual56logits, apply sigmoid and installed
   `paste_masks_in_image` unchanged at512. Gradients flow through sigmoid,
   bilinear interpolation, padding and CopySlices to the mask head.
3. Compute Euclidean2pixel bands on **annotations only**: inner foreground
   distance to background, and outer background distance to that instance.
   Image-frame edges are included in foreground distance. Other objects are
   negative relative to the matched owner; their masks are never merged.
4. Intersect bands with the original consumer's clipped/truncated half-open
   integer ROI. Compute separate mean BCE on foreground(FN) and background(FP),
   then average the available groups equally. Average proposals within each
   represented owner, then owners equally across the batch. Large background
   areas and objects with many proposals cannot dominate by pixel/proposal count.

The selected total is
`L_total = L_other + 2*L_ROI56 + .25*L_native`.
The auxiliary API weight is configurable/nonnegative; the **single admitted
proposal** fixes it to.25. Changing the proposed campaign requires rebinding and
review. Radius is fixed2native pixels; there is no search.

**No prediction thresholding, erosion, dilation or ANY2×2 enters autograd.**
Binary thresholding is used only by the future validation scorer. The band is a
fixed annotation selection; it does not modify training/inference predictions.
Outside-band interior foreground retains stock ROI56 supervision. Thin visible
features lie wholly in the foreground band; balanced FN penalties make collapse
worse than a correct mask in synthetic tests. This is not proof of trained
performance. Missing GT outside the consumer ROI has no optimisable mask gradient
and is counted as an outside-ROI sample diagnostic, not fed as a misleading
constant FN loss. Empty/unrepresentable integer ROIs or proposals with no band
intersection are explicitly counted/skipped; graph-connected zero handles empty
batches. Background images do not dilute the represented-owner mean.

BCE on probabilities uses Torch's finite endpoint handling. Extreme±100logits
have finite loss/gradients in tests; sigmoid saturation can yield zero gradients
and is **not** claimed to be corrected by this loss.

## Frozen state and synthetic verification

Only mask_head/mask_predictor parameters may require gradients. Frozen backbone
and non-mask BN/dropout are eval during the auxiliary forward; flags are restored.
All non-mask **parameters and buffers** are cloned/compared, not merely excluded
from SGD. The proposed worker checks exact frozen state after every update and
before publication; its optimiser contains only the12mask tensors.

Focused tests cover FP/FN gradient signs, collapse versus correct masks, balanced
owner/proposal normalisation, class selection, invalid ownership, occlusion,
thin/empty masks, bad boxes/nonfinite logits, extreme probabilities, exact
legacy dispatch, stock-loss tensor equality, actual head gradients, non-mask
buffers, deterministic repeated forwards, empty images, matcher ties, target
reordering, hook cleanup, gate materiality/no-harm and durable dummy supervision.
No real model is evaluated by these tests.

The bounded synthetic backward receipt records:

- Weighted auxiliary loss **.1687006801366806**; unweighted **.6748027205467224**.
- All **12mask parameter tensors** have finite, nonzero gradient L1 norms,
  including all four head convolutions and transpose/classifier parameters.
- Non-mask state hash before/after:
  `97543ba838472b7eb0de12c3ec57a08a630dd6a3f07fc5b9b2a4f2997bd5c17b`.
- All tensors unchanged; **0optimizer updates**, **0trained checkpoints loaded**.

Independent reviewer repeated the synthetic backward check with the same result.
Final test/resource counts and source/configuration binding are recorded
below. Evidence lives in
`evidence/segmentation_v2/native_boundary_implementation_001`; raw bounded test
logs are under`/home/user/epd-v2-artifacts/native-boundary-implementation-tests-001`.

## One proposed candidate training configuration

`tools/segmentation_v2/native_boundary_candidate.json` freezes:

| Setting | Value |
|---|---|
| Loss weights |ROI56=2; native boundary=.25; band radius2native pixels |
| Optimiser |Fresh SGD,LR.0001,momentum.9,weight_decay.0001 |
| Updates |One epoch,300updates,batch1,workers0,seed260110 |
| Splits |Immutable300training/12validation;12held-out excluded before file access |
| CPU/resources |2threads,10800s wall,6144MiB worker RSS,5GiB disk reserve |
| Output |`/home/user/epd-v2-artifacts/native-boundary-candidate-001`,must not exist |
| Retries/selection |No retries,sweeps,Stage-A2 selection,export or promotion |

Initializer is the byte-pinned **#221 epoch02** checkpoint
`f3e7531c25cb20920c04283d205d7241028d7321b1ea34e2f1848415744a3b56`, followed by
original`configure56`. Its recorded process exits0 without resource failure;
configuration/history/validation/checkpoint metadata are pinned. The future
worker additionally verifies epoch2/step600 and checkpoint history/validation
hash bindings after safe deserialization. This task only byte-hashed the
initializer; it did not load its weights.

This is **not continuation from** the missing-supervisor56checkpoint. Frozen
comparison baseline is that previous56campaign's retained12validation predictions,
whose checkpoint identity, source configuration, metrics and archive hashes are
pinned. Its training supervision remains UNKNOWN; it is comparison evidence,
not the candidate initializer or certified production provenance. Both recipes
start from #221, use the same56architecture/frozen non-mask state/seed/SGD/300steps;
the new intended change is consumer-space auxiliary supervision. Future paired
validation requires baseline/candidate boxes, labels and scores to be bitwise
identical, failing on non-mask drift.

**New native-paste backward runtime/memory overhead is UNMEASURED.** The3hour/6GiB
values are prospective hard caps, not extrapolated feasibility approval from the
old campaign or small unit model. If exceeded, stop and retain failure; no retry,
cap extension or replacement configuration is authorised here.

## Corrected durable supervisor

The prospective runner requires a separately authorised campaign flag and a
source/configuration-bound independent implementation review. It verifies29source
and9metadata evidence hashes, dataset/initializer/runtime and all frozen baseline
archive bindings before launch. It writes immutable`process_started.json` and
fsyncs`process.json:RUNNING` **before spawning**. A killed parent leaves an honest
incomplete record; no missing success record is manufactured.

The worker verifies live PPID, exact command/limits, admission hashes, pid/pgid,
new-session identity and Linux PDEATHSIG=SIGKILL before opening training inputs.
A bounded readiness handshake handles the parent writing worker identity.
Direct hidden-worker invocation is rejected. Shared supervision polls RSS every
20ms, checks wait4 high-water, enforces wall time, kills process groups, and
fsyncs logs. Atomic final process replacement records the **observed worker OS
exit**, measured wall/peak/limit reason. It does not fabricate the supervisor's
own unobserved OS exit. Synthetic dummy subprocesses test success, wall failure
and a positive live-launch handshake; existing tests cover RSS termination.

Source snapshots plus binary hashes, fsynced sequential history, frozen state,
validation/prediction hashes and atomic UNQUALIFIED checkpoint bind worker work.
Only the parent can publish a complete receipt after exit0/no-limit and checking
history/validation/checkpoint/frozen/source bindings. Partial data or a worker-only
result is never sufficient. The checkpoint remains UNQUALIFIED even on a
validation pass; independent training-provenance review is still required.
The original missing historicalprocess.json remains absent and unresolved.

## Candidate acceptance and remaining blockers

Future full12validation must show≥20%less background spill in **both** native512
consumer ROI and mapped256, with no per-instance/detection missing-foreground,
IoU, recall or cross-object regression, no per-scene spill/cross-object harm and
zero confident negative-scene detections. Frozen baseline assignment is preserved,
including duplicate/unassigned detections. Materiality across the full split
cannot hide sentinel harm: scenes024–027 also require native spill≤352vs441 and
mapped≤171vs214. All frozen non-mask state must match exactly. Any failed gate
rejects the candidate; no checkpoint selection search is implemented.

Native512-only supervision cannot directly resolve all independently rendered256
label disagreement or missing proposals. Those limitations are genuine; improved
loss/validation cannot be reported as EMD or Stage-A2 acceptance. Production
still requires genuine independent provenance qualification and strict Stage-A2
2/2 under unchanged rules, separately authorised and never used for tuning.

**Stop here:** implementation and synthetic validation only. The next decision is
whether to authorise **ONE bounded candidate training campaign** using this exact
reviewed configuration. No additional diagnostic investigation is proposed.


## Final verification and independent code review

**97/97 tests PASS**,0failures/errors,unittest2.004s. The final test worker ran
under120s/2GiB RSS/2CPU-thread limits: actual wall**5.404458436s**,peak**658896KiB**,
exit0,no limit reason. A separate bounded gradient worker completed in
**3.428882150s**,peak**643180KiB**,exit0; its12finite/nonzero gradient values are
retained in`gradient_verification.json`. The earlier test output001 and final002
are both preserved externally; no campaign directory was created.

Independent`/root/native_boundary_review` read the actual installed source,
reviewed all implementation/configuration/docs, independently passed18focused
synthetic tests, reproduced12mask tensor gradients and frozen state equality,
and verified a real disposable subprocess's live-parent/PDEATHSIG binding.
They made no implementation edits and ran no campaign or real inference.
Resolved findings: exact tie-safe matcher scope/empty lists; unmatched detection
collapse gate; fixed sentinel materiality; direct hidden-worker bypass. Decision
**APPROVE_IMPLEMENTATION_ONLY**,no unresolved findings,explicitly no training
execution or model qualification authorised.

Configuration SHA256:
`1b753bf1bca3a6466521459375c4d8f3e4cf900a6a905fc8750520a431263e6f`.
Genuine review SHA256:
`fcb9eb79b5f0255d95a006685b71bcb70b0a5f78d41aaa9d80474b6cd6f634c6`.
The configuration binds29source/runtime files and9metadata evidence records,
plus dataset/initializer and all12baseline archive hashes. All were verified
without deserialising trained weights or reading scene pixels in this task.
Historical R3/R4 output hashes were rechecked separately and remain unchanged.

Future command, **only after separate authorisation for one training campaign**:

```sh
python3 tools/segmentation_v2/native_boundary_candidate.py \
  --config tools/segmentation_v2/native_boundary_candidate.json \
  --review evidence/segmentation_v2/native_boundary_implementation_001/independent_review.json \
  --review-sha256 fcb9eb79b5f0255d95a006685b71bcb70b0a5f78d41aaa9d80474b6cd6f634c6 \
  --authorise-one-campaign
```

This command was **not executed**. The acknowledgement flag cannot substitute
for genuine human authorisation. Source/preregistration drift, missing review,
nonexistent live supervisor for worker calls, old output, resource failure or
validation regression all stop without automatic retry.
