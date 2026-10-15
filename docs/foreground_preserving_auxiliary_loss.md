# Foreground-preserving auxiliary revision 001

## Selected mechanism (recorded before implementation)

Keep the installed native512 differentiable sigmoid/paste, stock ROI56 loss,
consumer ROI, annotation boundary FP/FN terms and owner-balanced aggregation.
Add ONE foreground anchor: one-half the mean positive BCE over **all visible
owner foreground inside that proposal's valid consumer ROI**. The coefficient
one-half equals the existing boundary group coefficient when both groups exist;
it is fixed by this design, not selected by a sweep. Thus, per proposal:

`L_revision = L_previous_boundary + 0.5 * mean_FG(-log(p))`.

The auxiliary global weight remains0.25; radius remains2px. For one-pixel thin
objects all foreground already belongs to the inner band, so this doubles the
FN contribution when both boundary groups exist. For larger objects it adds
coverage of previously unpenalised interiors without diluting the old inner
boundary gradient. At the pasted probability level the new term has derivative
`-0.5/(N_FG*p)` for each valid foreground pixel, zero elsewhere (subject to the
installed BCE endpoint handling). The previous FP gradients are unchanged.
Every visible foreground pixel inside ROI therefore gets an additional
restorative gradient, and decreasing such probabilities cannot decrease this
anchor. Equal per-proposal/owner normalisation remains; no canvas-area or
proposal-count dominance. No amodal foreground is invented across occlusions.

This is a soft training objective, NOT a guarantee of non-regression after
shared-parameter optimisation: paste interpolation and shared weights couple
FP and FN effects. It can increase spill or reduce materiality. Frozen boxes
exclude outside-ROI truth; this term cannot recover missed detections or fix
native/mapped label disagreement. Exact original no-harm gates remain decisive.
No thresholds, morphology, baseline predictions or binary predicted masks enter
autograd. Weight0 dispatches directly to the legacy model forward.

## Evidence and scope

PR229's exact300-update candidate was rejected: native spill1193→888,
mapped647→527; missing5620→5846native and1183→1207mapped.17native/9mapped
instances lost foreground;6native/1mapped IoU regressed; recovery18/22 unchanged.
The retained scalar results and separately recorded pixel analysis support
increasing direct foreground coverage rather than accepting the FP/FN tradeoff.
Retained native masks show246previously covered foreground pixels lost and20
gained:190lost within2px of visible background,7at2–4px and49deeper
than4px. This supports both retaining boundary FN supervision and adding an
interior anchor. The per-instance pixel counts and archive/annotation hashes
are in `evidence/segmentation_v2/foreground_preserving_aux_001/retained_analysis.json`.
They do not prove a causal failure in the old loss or predict candidate success.
No new model inference or validation run is permitted in this implementation.
Thin-feature and occlusion-specific causes remain UNKNOWN unless identifiable
from the retained evidence; distance-to-visible-background alone identifies
neither feature thickness nor an occlusion boundary.

## Implementation and verification contract

Preserve the old module and candidate configuration unchanged. The revised
module preserves the reviewed capture/ownership path in a separate module;
there is no global monkeypatch or persistent inference hook. New candidate
identity/output are exclusive and source-bound. The rejected checkpoint remains
UNQUALIFIED and is never loaded here or selected as a future initializer.

Synthetic units must prove exact old FP derivatives, stronger foreground
restoration including interiors/thin/fragmented visible shapes, target ownership,
empty graph-connected zero, finite saturation handling, determinism and exact
zero-weight legacy terms. An invented4channel model with installed ROIHeads
must verify all12 mask tensors receive finite nonzero gradients and all nonmask
parameters/buffers stay unchanged. No optimizer step, trained weights or real
scene forward. Independent review is required before readiness for a separately
authorised campaign. Stage-A2 remains BLOCKED0/2; original supervision UNKNOWN.

## Retained per-instance foreground changes

These are baseline-covered native pixels lost minus newly covered pixels,
computed only from already retained probabilities, unchanged consumer ROI and
annotation ownership. They are not a new inference or evaluation campaign.
Distance bins measure distance inside the visible silhouette to background.

| Scene/owner | Lost | Gained | Lost ≤2px / 2–4px / >4px |
|---|---:|---:|---|
|scene_025/65508|28|0|26 / 2 / 0|
|scene_026/65496|24|3|11 / 1 / 12|
|scene_026/65501|15|0|6 / 2 / 7|
|scene_026/65506|15|15|13 / 1 / 1|
|scene_027/65510|15|0|15 / 0 / 0|
|scene_028/65503|11|1|11 / 0 / 0|
|scene_028/65508|7|0|7 / 0 / 0|
|scene_029/65496|11|0|10 / 1 / 0|
|scene_029/65506|16|1|11 / 0 / 5|
|scene_031/65508|14|0|14 / 0 / 0|
|scene_032/65496|32|0|8 / 0 / 24|
|scene_032/65501|12|0|12 / 0 / 0|
|scene_032/65506|4|0|4 / 0 / 0|
|scene_033/65510|12|0|12 / 0 / 0|
|scene_034/65503|5|0|5 / 0 / 0|
|scene_034/65508|12|0|12 / 0 / 0|
|scene_035/65501|7|0|7 / 0 / 0|
|scene_035/65506|6|0|6 / 0 / 0|

The scene026/65506 exchange15lost/15gained does not increase its missing total;
the other17objects have net missing regression.190/246lost pixels (77.24%)
are within2px;56/246 (22.76%) are outside the original FN band. No annotation
here establishes thin-feature or occlusion-specific cause: both remain UNKNOWN.

## Actual synthetic verification

Final bounded regression run: **118/118 PASS**,0failures/errors,unittest13.213s;
wall16.960523331s,peak669304KiB,workerexit0,no resource limit. This comprises
97historical regressions plus21revision/admission tests. No trained checkpoint,
real-scene forward, optimizer update or training was performed. Limits120s,
2GiB workerRSS,two CPUthreads. The first broad run had2fixture path errors when
run from tools/segmentation_v2; both new tests were corrected to resolve paths
relative to __file__. Failure logs are retained unchanged in tests-001;
corrected complete verification is tests-002. No loss/recipe changes were made
in response to these test-harness failures.

Separate bounded synthetic backward:wall3.800718133s,peak642956KiB,exit0.
All12mask parameter tensors received finite,nonzero gradients from the revised
auxiliary term alone (weightedloss0.25304198265075684). Nonmask gradients absent;
nonmask parameters/buffers and ALL model state unchanged (no optimizer step).
Before/after nonmaskSHA256:
`97543ba838472b7eb0de12c3ec57a08a630dd6a3f07fc5b9b2a4f2997bd5c17b`.
The network is an invented4channel/64internal-raster toy using installed
TorchvisionROIHeads/paste and native512synthetic targets; this proves graph
integration, not learned-model efficacy or full-model resource feasibility.

External evidence: `/home/user/epd-v2-artifacts/foreground-preserving-implementation-tests-001`
and `foreground-preserving-implementation-tests-002`; file hashes and gradient
report retained in `evidence/segmentation_v2/foreground_preserving_aux_001`.
Original PR229source/configuration and all bound inputs still hash-verify intact.

## Single preregistered candidate, NOT executed

`tools/segmentation_v2/foreground_preserving_candidate.json` is the separate,
immutable source-bound proposal; changing it requires new review/authorisation.
Configuration SHA256:
`55740d37fcf80d309379ac5c2f35f1c6deb8a344e22cc9ea028c22e98d1e09de`.
New exclusive futureoutput:
`/home/user/epd-v2-artifacts/foreground-preserving-candidate-001` (absent).
Fresh #221epoch02initializer SHA256
`f3e7531c25cb20920c04283d205d7241028d7321b1ea34e2f1848415744a3b56`,
adapted56-grid; never resume rejected checkpoint.300updates,seed260110,batch1,
freshSGD LR0.0001/momentum0.9/decay0.0001,ROIweight2,auxweight0.25,radius2,
fixedanchor0.5. Same immutable300train/12validation splits and frozen original
56-grid12prediction baseline; no held-out/Stage-A2 access. CPU2threads,
10800s,6144MiBworkerRSS,5GiBdiskreserve,onecampaign,no retry.

The separate runner retains the reviewed durable supervisor: fsynced RUNNING
receipt before spawn, exact liveparent/PID/PGID/command/config/review binding,
PDEATHSIG, resource enforcement, processgroup cleanup,durablelogs/history,
actual workerexit/wall/peakRSS and parentcompletion receipt. Polling+wait4memory
enforcement is not a cgroup hard cap. Admission rejects direct hiddenworker,
source/loss identity drift, stale/nonexclusive output or absent authorisation.

Evaluation gates are byte-for-byte equal to PR229configuration:full12validation,
≥20% BOTH native512/mapped256 spill reduction; no per-scene spill/cross/recall
harm; no per-instance OR detection missing/IoU/cross regression;zero confident
negative-sceneFPs;exact frozen nonmaskstate and independently verified provenance.
Fixed024–027sentinelcaps remain native≤352/mapped≤171. Any failure rejects.
No threshold/geometry/association change; Stage-A2 remainsBLOCKED0/2;
original training supervisionUNKNOWN; rejected checkpoint staysUNQUALIFIED.

Only after separate explicit authorisation, the proposed command is:

```sh
python3 tools/segmentation_v2/foreground_preserving_candidate.py \
  --config tools/segmentation_v2/foreground_preserving_candidate.json \
  --review evidence/segmentation_v2/foreground_preserving_aux_001/independent_review.json \
  --review-sha256 23bd4936756b5cb46a66f27ff67dba39108d7fffe716c6f78f235c9d4483c8c1 \
  --authorise-one-campaign
```

This command has NOT been executed. The CLIflag is not human consent.
The next decision is whether to authorise ONE bounded training campaign.

## Independent implementation review and readiness

Independent `/root/foreground_revision_review` approved IMPLEMENTATION_ONLY,
no unresolved findings. They independently passed12auxiliary and9candidate
tests, verified all12finite/nonzero gradients and unchanged state, checked
all35source+14evidence bindings, and reviewed the unchanged gates/recipe and
the foreground/BG derivative mechanism. Source-bound approval:
`evidence/segmentation_v2/foreground_preserving_aux_001/independent_review.json`,
SHA256`23bd4936756b5cb46a66f27ff67dba39108d7fffe716c6f78f235c9d4483c8c1`.
Approval is implementation-only; it cannot authorise training or qualify a model.

**Ready for separate authorisation of ONE bounded campaign**, subject to the
same clean-tree/hash/review/exclusive-output/resource admission checks at launch.
No production improvement is claimed. Primary remaining risks are increased
spill from stronger FG pressure, shared-parameter pixeltradeoffs, saturation,
ROI-excluded foreground, fixed proposal recall and native/mapped raster
inconsistency. Any original acceptance failure rejects the next candidate.
