# Analytic ROIAlign→synthetic56→paste coordinate contract

This CPU-only benchmark follows draft #226's measured spill/completeness
tradeoffs. It loads no checkpoint, trains no model and reads no real scene,
renderer label or held-out/Stage-A2 pixel. Stage-A2 remains BLOCKED0/2 and original
training supervision UNKNOWN. No production geometry change is authorised.

## Preregistered methodology

Six shapes: circle, rectangle,30-degree rotated rectangle, triangular sharp
edge,1.5px thin bar and empty. Analytic geometry lives in continuous image-edge
coordinates [0,128]²; image samples are at(j+.5,i+.5). Strict membership is
computed independently from model probabilities and from the signed-distance
implementation used to construct the input field. The input scalar field is
exact positive-inside Euclidean signed distance in pixels (empty=-20), plus
world-x/world-y ramp channels to expose coordinate placement.

Box sizes12×10,32×28,64×52; offsets(0,0),(.25,.75),(.5,.5); placements interior,
top-left clipped and bottom-right clipped. Analytic shapes remain defined in
the unclipped box while the operator box is image-clipped. Spatial scales1,.5,.25
and two declared feature-grid origins yield **972 feature settings**:

- `cell_centred`: world sample coordinate(j+.5)/scale.
- `fixed_input_centre`: world sample coordinate.5+j/scale.

These are **synthetic coordinate assumptions, not established trained FPN
origins**. The benchmark deliberately tests both rather than assuming one.

Actual compiled Torchvision ROIAlign pools28×28, sampling_ratio2, with legacy
aligned=False and the explicit aligned=True shadow. It is independently
reimplemented in NumPy float32 using the pinned CPU source's sample positions,
bilinear weights, boundary clamping and sequential accumulation. Every pooled
value must match **bitwise** before any alternatives are evaluated.

The learned mask head is **not executed or emulated**. Two declared synthetic
bridges produce56×56 logits: bilinear28→56 (`align_corners=False`) and2×2
replication (bitwise equal to a unit-kernel stride2 transposed convolution).
They cannot represent learned convolutions/nonlinearities or prove information
loss in the actual trained head. There are **1944 round-trip cases** and **7776
arm measurements**, all reported without choosing a production winner.

Four arms form a2×2 factor design:

| Arm | ROIAlign | Paste |
|---|---|---|
| Legacy | aligned=False | Original padded/expanded/truncated/inclusive Torchvision |
| Continuous paste only | aligned=False | Unrounded pixel-centre/bin-centre mapping |
| Aligned ROI only | aligned=True | Original Torchvision |
| Aligned + continuous | aligned=True | Unrounded mapping |

The paired arm changes two conventions; effects are not additive causal
partitions. Original paste must also match the independent reconstruction
bitwise, for both ROIAlign flags and both bridges, before shadow comparisons.
Continuous paste samples sigmoid logits via bilinear grid_sample with zeros
outside and `align_corners=False`, using `u=56*(x+.5-x0)/width-.5`. It uses no
reference mask/label to trim output. Independent NumPy interpolation tests check
that mathematical definition. No erosion/dilation or production replacement.

### Measurements and uncertainty

Report centre-sampled IoU, spill, missing, foreground counts, conservation
`pred-reference=spill-missing`, centroid offsets, and signed/absolute analytic
boundary displacement. Boundary locations are interpolated0.5 crossings along
adjacent pixel centres, evaluated against the exact geometry's signed distance.
Positive displacement means outward. **Image-frame boundary edges are omitted**;
statistics are not full Hausdorff distance. Extinguished shapes have absent
contours and are counted explicitly, not interpreted as zero boundary error.

The conservative raster uncertainty band is `abs(exact SDF)<=sqrt(.5)` pixels,
the half-diagonal of a unit cell. Errors outside it cannot be explained solely
by choosing a different point within the same pixel cell.4×4 and8×8 subpixel
membership estimates within that band provide approximate coverage-area/IoU
and convergence diagnostics. **Their differences are not guaranteed error bars**.
Centre counts, coverage estimates and band-exceeding errors are all reported.

72 preregistered traces (size32×28,offset.25/.75,interior,all shapes/scales/origins/
bridges) retain feature fields, actual sample grids, pooled28, synthetic logits56,
coordinate ramp fields, expanded float/truncated boxes, inclusive paste sizes,
reference coverage and all four probability outputs for independent replay.

### Competing preregistered explanations

1. Feature-origin mismatch: on unclamped interior ramps, legacy ROIAlign is
   expected to sample at canonical bin centres plus the chosen origin. Under
   cell-centred fields that is+.5/scale pixels; aligned=True removes that offset.
   Under fixed-input-centre fields the corresponding offsets are+.5 and
   `.5-.5/scale`. Thus no aligned flag is universally justified by its name.
2. Legacy paste quantization/padding/inclusive extents add box-phase-dependent
   effects; a global half-pixel shift need not repair the whole chain.
3. Coarse feature sampling,28-bin averaging and the declared synthetic bridge
   can lose thin-feature information even with coordinate-consistent pasting.
4. Centre-sampled raster truth, boundary clipping and interpolation of nonlinear
   fields can produce differences that are not coordinate-contract defects.

## Source and review binding

`analytic_coordinate_preregistration.json` SHA256:
`a2ded7d5082d12ea3612e5d4ef346f01e607b942177c82db688b3872feb0778d`.
Executed benchmark source SHA256:
`7c35607cd4ebea5e2a7d676decac6b1c8e60ba81bfce9f5d79f47823088a57c9`.
The original #226 runtime/source inventory is preserved and checked exactly:
Torch2.7.0,Torchvision0.22.0,NumPy1.26.4,OpenCV4.11.0.86, including compiled
Torchvision/Torch binary hashes. No library install/update occurs.

Pinned official CPU source snapshots and BSD license are retained in
`evidence/segmentation_v2/analytic_coordinate_sources`. References:
[ROIAlign CPU kernel](https://raw.githubusercontent.com/pytorch/vision/v0.22.0/torchvision/csrc/ops/cpu/roi_align_kernel.cpp),
[bilinear sample-grid helper](https://raw.githubusercontent.com/pytorch/vision/v0.22.0/torchvision/csrc/ops/cpu/roi_align_common.h).
Their SHA256 values are
`d3a2f0d909d9a7bb47c30cd82058a4afb02453498e3283f62dbf82a64e65a769`
and `1e22a97d9b3c84f6c9782fa4db0cb27d03cfd760fa5aa736413df2bc046e724e`.
Snapshots identify reviewed source; exact runtime parity identifies observed
operator behavior without claiming independent binary-build attestation.

Independent methodology reviewer `/root/diagnostic_admission_review` approved
only this synthetic methodology after the interpretation limits were explicitly
bound into policy/results. No actionable findings remained. Review SHA256:
`45a062f70535d7a243f1d7339b78e8a6a744577cadd93824728286372177114f`.
The review and test log are retained in the new benchmark evidence directory.

Before execution **60/60 focused tests PASS**,0 failures/errors, unittest1.060s,
whole process4.38s,peak619308KiB. Ten new tests cover exact ROI/paste parity,
tiny/degenerate/fractional/clipped boxes, independent shape membership/SDF,
uncertainty, pixel-centre offsets, unit deconvolution, strict thresholds,
immutability, extinguished features and exclusive output. Existing shared
supervisor tests exercise actual dummy wall/RSS termination and preserved logs.
Broad training/model experiments are not run.

The new exclusive benchmark output is supervised at300s,2GiB worker RSS and2
CPU threads, with parent-death signalling, process-group cleanup and durable
logs. RSS uses20ms polling plus wait4 high-water rejection, not kernel memory.max.
Partial/failure evidence is preserved; there is no automatic retry.

## Actual single attempt: BLOCKED_FAILURE

The preregistered attempt was launched once from commit
`51bb1d4a5ac27c288b166923e7d9a22d3345e48a` into the new exclusive directory
`/home/user/epd-v2-artifacts/analytic-coordinate-contract-001`:

```sh
python3 tools/segmentation_v2/analytic_coordinate_contract.py \
  --policy tools/segmentation_v2/analytic_coordinate_preregistration.json \
  --output /home/user/epd-v2-artifacts/analytic-coordinate-contract-001
```

**The benchmark did not complete.** Exit1 after13.17655985s, peak worker RSS
1,039,084KiB; neither300s nor2GiB limit was reached. Process-group cleanup and
parent-death SIGKILL were configured/recorded. Original logs, partial JSONL,
launch identity, failure and supervision records remain preserved. No retry.

Before any shadows, the entire original-operator gate completed:

| Bitwise comparison | Checks | Maximum absolute difference |
|---|---:|---:|
| Independent NumPy ROIAlign vs compiled Torchvision | 1,944 | 0 |
| Independent original paste vs Torchvision | 3,888 | 0 |
| Replication vs unit2×2 stride2 deconvolution | 1,944 | 0 |

Trace serialization then raised
`TypeError: unsupported operand type(s) for *: 'numpy.ndarray' and 'Tensor'`.
Only145/1,944 round-trip rows (580 arm measurements) were retained, all circles;
108 are size12×10 and37 size32×28. All580 conservation checks pass. **Zero trace
archives and no final results.json exist**. Empty, rotated, sharp-edge,
rectangle and thin-feature round-trip findings therefore remain unmeasured.
The final source-recheck/publication stage was not reached. The parity receipt
is complete, but does not turn this failed attempt into a completed experiment.

The executed source is retained byte-for-byte in
`evidence/segmentation_v2/analytic_coordinate_benchmark_001/executed_analytic_coordinate_contract.py`
with SHA256`7c35607cd4ebea5e2a7d676decac6b1c8e60ba81bfce9f5d79f47823088a57c9`.
The serialization correction now converts immutable tensor endpoints explicitly
before NumPy arithmetic. A regression checks exact56-bin centres, immutable
boxes and actual NPZ serialization. **61/61 focused tests PASS**,0 failures/errors,
unittest1.102s, process4.43s,peak619284KiB. Corrected source SHA256:
`00fdd8969909be5d091e746cc8eb19796299b145a03a3c6c3c5b917ff47db4f4`.
The original preregistration remains unchanged: its source binding rejects the
corrected implementation. Updating/reviewing a future preregistration and a
separately authorised new attempt are required; no automatic retry was made.

### Partial observations, not full benchmark conclusions

Offline analysis selects the complete size12×10/interior circle subset:
three offsets×two bridges per origin/scale (six cases, **not independent
replicates**). Its ramp errors in each coordinate agree with the preregistered
formula to roughly2e-6px:

| Synthetic feature origin | Scale | Legacy ROI offset(px) | Aligned ROI offset(px) |
|---|---:|---:|---:|
| cell_centred | 1 | +0.5 | 0 |
| cell_centred | .5 | +1 | 0 |
| cell_centred | .25 | +2 | 0 |
| fixed_input_centre | 1 | +0.5 | 0 |
| fixed_input_centre | .5 | +0.5 | −0.5 |
| fixed_input_centre | .25 | +0.5 | −1.5 |

For cell-centred scale1 circles, paired alignment+continuous paste improves
mean IoU and reduces measured displacement; at scale.25 significant missing
area remains even with zero spill. Under the other declared origin, the paired
arm performs worse at scale.25. Neither origin has been established for trained
FPN features.

| Origin/scale | Arm | Mean IoU | Spill sum | Missing sum | Mean absolute contour displacement(px) | Errors outside sampling band |
|---|---|---:|---:|---:|---:|---:|
| cell/1 | original | .732656 | 44 | 13 | .538357 | 4 |
| cell/1 | continuous paste only | .739973 | 26 | 25 | .438922 | 0 |
| cell/1 | aligned ROI only | .875412 | 21 | 3 | .240901 | 0 |
| cell/1 | paired | .994253 | 1 | 0 | .037208 | 0 |
| cell/.25 | original | .226602 | 68 | 116 | 1.571222 | 116 |
| cell/.25 | continuous paste only | .238650 | 56 | 116 | 1.391699 | 106 |
| cell/.25 | aligned ROI only | .759286 | 0 | 41 | .429409 | 4 |
| cell/.25 | paired | .670000 | 0 | 58 | .597816 | 8 |
| fixed/.25 | original | .502563 | 15 | 77 | .745400 | 41 |
| fixed/.25 | continuous paste only | .490246 | 4 | 85 | .794352 | 40 |
| fixed/.25 | aligned ROI only | .316427 | 39 | 104 | 1.196728 | 79 |
| fixed/.25 | paired | .354228 | 22 | 102 | 1.082766 | 62 |

Displacement is the mean of six case means, not a pooled Hausdorff statistic.
Image-frame edges are omitted; the selected cases are interior. The conservative
sampling band is±sqrt(.5)px; coverage4/8 are approximate quadratures, not error
bars. Counts repeat the same geometry under offsets/bridges. This is descriptive
analysis of a failed prefix, not a selected alternative or production claim.

**Observations:** exact original operator parity; origin-dependent ramp offsets;
paste convention effects interact with sampling scale, foreground loss and IoU.
**Competing hypotheses still open:** actual feature-lattice origin, nonlinear
field interpolation, coarse sampling/28→56 synthetic bridge loss, box rounding
and inclusive extent, and learned head compensation. The synthetic bridges
cannot establish learned-head information loss or attribute #226's real-scene
contamination. The missing shape/trace matrix prevents broader conclusions.

### Retained evidence and independent review

The offline-only script below recomputes partial summaries, conservation checks
and the complete attempt artifact manifest, without running any operator:

```sh
python3 evidence/segmentation_v2/analytic_coordinate_benchmark_001/analyse_partial.py \
  /home/user/epd-v2-artifacts/analytic-coordinate-contract-001
```

Retained `partial_analysis.json` SHA256:
`611251b8f93dd3d7e5a073a39fe550f9d115307a88c85a4bb2ed6d1d3aff420e`.
External measurements SHA256:
`7512b2ced4b090def6b3ff6d3058eb5a90dab467a500af1810e341d6457f9ccf`.
Parity receipt SHA256:
`8361ab9f53ff5801a702359fa7733f1bdb5330054c83c9aeb808a1da8a69aee3`.
Supervision SHA256:
`c779221fa175172178cb1aae2d46cf1fd0a67b7f0a154244ca67349062be5410`.

Independent reviewer`/root/diagnostic_admission_review` verified failure evidence,
parity receipt/counts/code ordering, all partial conservation checks, absent
traces/results, executed-source retention and correction. They independently
passed61 tests; they did not rerun the benchmark. Their decision is
`FAILED_SINGLE_ATTEMPT_PARTIAL_EVIDENCE_ONLY`, with
`SERIALIZATION_FIX_ACCEPTED_NO_RERUN`. Review SHA256:
`c04c471959ca0fd28a912b6be8b4b0c74d2c811ddc27b7f2a5243202f2d90d1c`.
Both reviews, source snapshots, test logs, copied failure receipts and offline
analysis are committed; original external evidence is unchanged.

**One next milestone:** independently rebind/review the serialization-corrected
preregistration and separately authorise one new exclusive synthetic run to
complete the existing matrix. No new arms, model experiment or production
geometry change is supported by this incomplete attempt. Stage-A2 remains
BLOCKED0/2; original training supervision remains UNKNOWN.

## Authorised R3 attempt (new binding)

The user separately authorised one corrected synthetic attempt in exclusive
`/home/user/epd-v2-artifacts/analytic-coordinate-contract-002`. The001 failure is
preserved. Numerical core source remains`00fdd8969909be5d091e746cc8eb19796299b145a03a3c6c3c5b917ff47db4f4`;
only a separate R3 admission/completeness wrapper is added. The immutable new
preregistration is`tools/segmentation_v2/analytic_coordinate_preregistration_r3.json`,
SHA256`191b1519907dcd066a5c99317b4403161a1db32d2453dcc1ec2e1f74a68c5a23`.
It retains the exact geometry matrix/operators/threshold, binds corrected
source/tests/runtime, and requires1944cases,7776arm measurements,72 fully shaped
finite trace archives, all original parity checks, exact case identities,
conservation, recomputed summaries and source integrity. Missing/partial/drifted
outputs fail closed. The wrapper checks independent source-bound approval
before creating any output, and supervises numerical work plus validation under
300seconds,2GiB worker RSS and2CPU threads. No retries.

Independent reviewer`/root/r3_review` verified the correction leaves numerical
controls unchanged, all nine001 external artifact hashes remain intact, and
all R3 bindings match. Their finding that every trace field needed exact shape
validation was fixed and regression-tested before approval. Decision:
`APPROVE_R3_SYNTHETIC_ONLY`; review SHA256
`f14fd7a37c447317de9f84f1985e5d4afe7c700f2ffe141c00d04fd45de7dda6`.
Review/test receipts are in`evidence/segmentation_v2/analytic_coordinate_benchmark_002`.
Pre-run focused tests:**67/67 PASS**,0failures/errors,unittest1.386s,
process4.63s,peak621764KiB; reviewer independently ran17analytic/R3 tests.

```sh
python3 tools/segmentation_v2/analytic_coordinate_r3.py \
  --policy tools/segmentation_v2/analytic_coordinate_preregistration_r3.json \
  --review evidence/segmentation_v2/analytic_coordinate_benchmark_002/prerun_review.json \
  --review-sha256 f14fd7a37c447317de9f84f1985e5d4afe7c700f2ffe141c00d04fd45de7dda6
```
