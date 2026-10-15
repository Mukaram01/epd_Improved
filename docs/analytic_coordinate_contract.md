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
