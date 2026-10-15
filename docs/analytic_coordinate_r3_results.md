# R3 completed synthetic coordinate-contract benchmark

## Decision and limits

**COMPLETE_SYNTHETIC_ONLY:** 1,944/1,944 round-trip cases, 7,776 arm measurements,
72/72 complete trace archives, all six shapes and every preregistered size,
offset, clipping placement, scale, origin and bridge. One new attempt ran; the
previous failed001 attempt is unchanged. Stage-A2 remains **BLOCKED0/2** and
original training supervision **UNKNOWN**. No trained-model qualification or
production change follows from this experiment.

Source-bound independent pre-run approval and post-run verification were by
`/root/r3_review`, distinct from implementer`/root`. The reviewer required exact
shapes for all trace fields; this was fixed/tested before approval. They later
verified complete identities, traces, source hashes, original failure hashes
and independently reproduced3,776 scalar/array summary checks from raw rows
within1e-12 tolerance, without repeating the benchmark. No actionable findings.

**Tests:67/67 PASS**,0failures/errors; unittest1.386s,whole process4.63s,
peak621764KiB. Coverage includes corrected trace serialization, immutable
boxes, original numerical parity, tiny/degenerate/clipped/subpixel inputs,
strict>0.5 threshold, missing/duplicate cases, malformed trace shape, invalid
numbers, conservation, input/hash drift, independent review, output exclusivity
and actual dummy worker wall/RSS termination with failure evidence preservation.

## Runtime and operator parity

The exact command and preregistration are recorded in the main coordinate
contract document. Execution began from commit`0de9414` after independent
approval. CPU-only, two CPU threads,300s wall and2GiB worker RSS limits.
Supervisor exit0, **22.288214992s**, peak worker RSS **1,037,020KiB**, resource-limit
reason`null`, process-group cleanup true, parent-death SIGKILL. Numerical worker
measurement16.489857574s excludes startup/final completeness validation.
RSS enforcement is20ms polling plus wait4 high-water rejection, not kernel
cgroup memory.max. Durable stdout/stderr are empty; no retry occurred.

| Original operator comparison | Bitwise checks | Maximum absolute difference |
|---|---:|---:|
| Independent NumPy ROIAlign vs compiled Torchvision, both flags | 1,944 | 0 |
| Independent original paste vs Torchvision, both bridges/flags | 3,888 | 0 |
| Replication vs unit2×2 stride2 transposed convolution | 1,944 | 0 |

All originals passed before any shadow comparison. The synthetic bilinear and
unit-deconvolution bridges are explicit fields, **not trained mask-head outputs**.
The original production operators, weights, thresholds and EMD are unchanged.

## Mathematically supported displacement

World samples are defined either at`(j+.5)/scale` or`.5+j/scale`. On interior
coordinate ramps, the actual ROI grid matches the predicted offsets below in
both x and y. Maximum residual of any case-mean ramp offset is
**2.093e-6px** (float32 accumulation). These90 nonempty round-trip cases per
origin/scale repeat geometry across shapes and bridges; they are not independent
statistical replicates. Positive ROI sampling displacement generally moves
reconstructed contours in the opposite direction under inverse pasting.

| Feature-origin assumption | Scale | Legacy sample displacement x,y(px) | Aligned sample displacement x,y(px) |
|---|---:|---:|---:|
| cell_centred | 1 | +0.5, +0.5 | +0, +0 |
| cell_centred | 0.5 | +1, +1 | +0, +0 |
| cell_centred | 0.25 | +2, +2 | +0, +0 |
| fixed_input_centre | 1 | +0.5, +0.5 | +0, +0 |
| fixed_input_centre | 0.5 | +0.5, +0.5 | -0.5, -0.5 |
| fixed_input_centre | 0.25 | +0.5, +0.5 | -1.5, -1.5 |

This establishes systematic coordinate-origin sensitivity for the specified
analytic fields. It **does not establish the actual trained FPN lattice origin**
or prove that aligned=True is a general correction. Learned features can have
architectural or learned compensation absent from these fields.

## Original versus fixed shadow operators

Each origin/scale row aggregates270 nonempty round-trip cases over all five
nonempty shapes, sizes, offsets, placements and both bridges. IoU is an
unweighted case mean. Boundary error is absolute analytic SDF displacement at
interpolated0.5 crossings, weighted by crossing counts; image-frame edges are
omitted. Extinguished cases have no contours and are counted explicitly.
Counts sum pixels across repeated synthetic cases; they are not unique scene
pixels. Paired changes both factors, so differences are not additive attribution.

| Origin | Scale | Arm | Mean IoU | Spill | Missing | Foreground delta | Extinguished | Boundary error(px) | Errors outside±.707107px band |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cell_centred | 1 | Original | 0.795663 | 4335 | 2749 | +1586 | 15 | 0.386109 | 695 |
| cell_centred | 1 | Continuous paste | 0.753425 | 3738 | 4729 | -991 | 14 | 0.464938 | 482 |
| cell_centred | 1 | Aligned ROI | 0.814308 | 3822 | 2668 | +1154 | 15 | 0.350374 | 467 |
| cell_centred | 1 | Paired | 0.907577 | 829 | 1955 | -1126 | 16 | 0.109515 | 89 |
| cell_centred | 0.5 | Original | 0.640222 | 6979 | 6245 | +734 | 34 | 0.761064 | 3736 |
| cell_centred | 0.5 | Continuous paste | 0.602377 | 6925 | 8537 | -1612 | 35 | 0.905919 | 4957 |
| cell_centred | 0.5 | Aligned ROI | 0.763700 | 3415 | 3148 | +267 | 33 | 0.352608 | 562 |
| cell_centred | 0.5 | Paired | 0.807054 | 794 | 2668 | -1874 | 34 | 0.129755 | 186 |
| cell_centred | 0.25 | Original | 0.475837 | 12708 | 13490 | -782 | 49 | 1.631020 | 15064 |
| cell_centred | 0.25 | Continuous paste | 0.459527 | 12693 | 15661 | -2968 | 51 | 1.767611 | 16957 |
| cell_centred | 0.25 | Aligned ROI | 0.653366 | 3128 | 4734 | -1606 | 49 | 0.392299 | 1273 |
| cell_centred | 0.25 | Paired | 0.659715 | 1107 | 4750 | -3643 | 51 | 0.222616 | 968 |
| fixed_input_centre | 1 | Original | 0.795663 | 4335 | 2749 | +1586 | 15 | 0.386109 | 695 |
| fixed_input_centre | 1 | Continuous paste | 0.753425 | 3738 | 4729 | -991 | 14 | 0.464938 | 482 |
| fixed_input_centre | 1 | Aligned ROI | 0.814308 | 3822 | 2668 | +1154 | 15 | 0.350374 | 467 |
| fixed_input_centre | 1 | Paired | 0.907577 | 829 | 1955 | -1126 | 16 | 0.109515 | 89 |
| fixed_input_centre | 0.5 | Original | 0.741584 | 4045 | 3259 | +786 | 35 | 0.391145 | 760 |
| fixed_input_centre | 0.5 | Continuous paste | 0.696928 | 3495 | 5177 | -1682 | 40 | 0.470906 | 618 |
| fixed_input_centre | 0.5 | Aligned ROI | 0.651312 | 6242 | 5799 | +443 | 37 | 0.691483 | 3077 |
| fixed_input_centre | 0.5 | Paired | 0.701083 | 3252 | 5213 | -1961 | 38 | 0.473370 | 660 |
| fixed_input_centre | 0.25 | Original | 0.645160 | 4060 | 4760 | -700 | 47 | 0.435394 | 1498 |
| fixed_input_centre | 0.25 | Continuous paste | 0.625812 | 3546 | 6446 | -2900 | 50 | 0.509601 | 1639 |
| fixed_input_centre | 0.25 | Aligned ROI | 0.465138 | 11924 | 13773 | -1849 | 46 | 1.539401 | 14872 |
| fixed_input_centre | 0.25 | Paired | 0.493847 | 9217 | 13165 | -3948 | 50 | 1.341075 | 11640 |

For cell-centred scale¼, paired IoU improves.475837→.659715 and absolute
boundary error1.631020→.222616px; yet51/270 shapes disappear versus49/270
originally. Under fixed-input-centre scale¼, paired IoU worsens.645160→.493847
and boundary error.435394→1.341075px. Thus spill reduction alone cannot select
a geometry change. Across all1,620 nonempty cases, original spill/missing is
36,462/33,252 (delta+3,210), paired16,028/29,706 (delta−13,678). Every one of
7,776 arm records satisfies `pred-reference = spill-missing`.

The following signed x/y values are **predicted-minus-reference pixel-centre
centroids**, averaged only over cases with both centroids. They mix shift,
shape deformation and clipping, unlike the pure coordinate ramps. Boundary
signed mean is outward-positive and crossing-weighted.

| Origin | Scale | Arm | Centroid dx,dy(px) | Centroid cases | Signed boundary mean(px) |
|---|---:|---|---:|---:|---:|
| cell_centred | 1 | Original | -0.296279, -0.299543 | 255 | +0.132437 |
| cell_centred | 1 | Paired | -0.006337, +0.009385 | 254 | -0.043785 |
| cell_centred | 0.5 | Original | -0.762462, -0.767702 | 236 | +0.181596 |
| cell_centred | 0.5 | Paired | +0.020616, +0.012632 | 236 | -0.064608 |
| cell_centred | 0.25 | Original | -1.628725, -1.648697 | 221 | +0.283529 |
| cell_centred | 0.25 | Paired | +0.071503, +0.029636 | 219 | -0.159556 |
| fixed_input_centre | 1 | Original | -0.296279, -0.299543 | 255 | +0.132437 |
| fixed_input_centre | 1 | Paired | -0.006337, +0.009385 | 254 | -0.043785 |
| fixed_input_centre | 0.5 | Original | -0.261236, -0.281219 | 235 | +0.116544 |
| fixed_input_centre | 0.5 | Paired | +0.515792, +0.502852 | 232 | -0.076452 |
| fixed_input_centre | 0.25 | Original | -0.175256, -0.224779 | 223 | +0.029205 |
| fixed_input_centre | 0.25 | Paired | +1.528968, +1.498591 | 220 | -0.122278 |

## Geometry and discrete-resolution findings

All324 empty cases produce zero foreground in every arm, with IoU1 and no
contour. Each other shape also has324 round-trip cases. Full shape×origin×scale
metrics are retained in`offline_summary.json:shape_origin_scale`.

| Shape | Original IoU | Continuous paste IoU | Aligned ROI IoU | Paired IoU | Original spill/missing | Paired spill/missing | Original→paired extinguished |
|---|---:|---:|---:|---:|---:|---:|---:|
| circle | 0.802973 | 0.785365 | 0.820110 | 0.872507 | 7252/5414 | 2875/4726 | 0→0 |
| empty | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0/0 | 0/0 | 0→0 |
| rectangle | 0.817021 | 0.799581 | 0.836846 | 0.874638 | 10283/7905 | 4371/6765 | 0→0 |
| rotated_rectangle | 0.819029 | 0.804116 | 0.821171 | 0.869931 | 7319/5674 | 2968/4907 | 0→0 |
| sharp_edge | 0.781661 | 0.722203 | 0.767667 | 0.836152 | 10729/8376 | 5458/7599 | 2→2 |
| thin_feature | 0.191092 | 0.131648 | 0.222651 | 0.277482 | 879/5883 | 356/5709 | 193→203 |

For the1.5px thin bar under cell-centred scale¼, **20/27 unique feature
settings** contain no positive sampled SDF value;23/27 have no positive pooled
ROI bin under either alignment. Both synthetic bridges then remain nonpositive
in those23 settings. These are positivity/completeness losses in the declared
fixed pipeline; adjusting paste coordinates cannot restore a>0.5 foreground
from wholly nonpositive logits. They do not prove geometry is unrecoverable by
an arbitrary estimator from real-valued samples.

Across all nonempty cases, bilinear28→56 extinguishes all positive logits in
**16 cases per ROI flag** despite positive pooled bins. Replication does so in
**0** cases, as required by its mathematical definition. This separates a
synthetic bridge averaging effect from coordinate offset. The coarse feature
sampling, ROI averaging, bridge and final raster threshold each contribute;
these counts are conditional stage observations rather than additive causal
percentages. The learned mask head is not emulated.

The fixed size/offset/clipping/bridge marginals below expose phase and border
sensitivity. Each size/offset/placement aggregates540 nonempty cases; each
bridge810. These mix origins/scales: use the full raw matrix for conditional
comparisons, not these marginals to choose an alternative.

| Dimension | Value | Original IoU | Continuous paste IoU | Aligned ROI IoU | Paired IoU |
|---|---|---:|---:|---:|---:|
| size | [12, 10] | 0.578893 | 0.527017 | 0.593316 | 0.674378 |
| size | [32, 28] | 0.727263 | 0.692491 | 0.744584 | 0.798028 |
| size | [64, 52] | 0.740909 | 0.726240 | 0.743166 | 0.766020 |
| offset | [0.0, 0.0] | 0.679212 | 0.644622 | 0.692301 | 0.746774 |
| offset | [0.25, 0.75] | 0.679451 | 0.643370 | 0.692213 | 0.741463 |
| offset | [0.5, 0.5] | 0.688402 | 0.657755 | 0.696552 | 0.750189 |
| placement | border | 0.730489 | 0.638560 | 0.667933 | 0.744979 |
| placement | far_border | 0.670657 | 0.669220 | 0.699908 | 0.752700 |
| placement | interior | 0.645918 | 0.637968 | 0.713226 | 0.740746 |
| bridge | bilinear | 0.676934 | 0.644649 | 0.692378 | 0.745667 |
| bridge | replicate | 0.687776 | 0.652516 | 0.694999 | 0.746617 |

The conservative raster-sampling band is`abs(SDF)<=sqrt(.5)`px. Errors outside
it cannot be explained solely by choosing another point within a pixel cell.
Coverage4×4 versus8×8 area estimates differ by mean.595775px² and maximum
4.5625px² per case. These are convergence diagnostics, **not guaranteed error
bars**. Boundary sampling omits frame edges and is not full Hausdorff distance.
Aggregate centroid/contour statistics omit absent contours/centroids; IoU,
missing and extinction counts prevent interpreting disappearance as improvement.

## Evidence, reproducibility and independent review

Output directory:`/home/user/epd-v2-artifacts/analytic-coordinate-contract-002`.
All raw measurements,72 traces, logs, launch identity, source inventory, result,
validation and supervision receipts are retained there. Committed lightweight
receipts/summary/reviews reside in
`evidence/segmentation_v2/analytic_coordinate_benchmark_002`.

| Binding | SHA256 |
|---|---|
| R3 preregistration | `191b1519907dcd066a5c99317b4403161a1db32d2453dcc1ec2e1f74a68c5a23` |
| Corrected numerical core | `00fdd8969909be5d091e746cc8eb19796299b145a03a3c6c3c5b917ff47db4f4` |
| R3 admission/completeness wrapper | `e83ddfa93451f7aace19f1e0e0c68063c5855362e8b08b34aa8f13f738814a06` |
| Results | `a25f734f45e5cdf2ef9ff05e95e2fb664da70b78e0f5c265aef164cf1687f6ad` |
| Raw measurements | `ed1f14c307f8791feb74b39a967fb85dc6cee882b1045a4f4dd1f6695ac37138` |
| Supervisor receipt | `2d95f76bb320b93c906e691f9aa2a35e4e207b0f42a7ccc622f07d24db2e9824` |
| Offline summary/full artifact manifest | `b6b84e5a5c0cbc4f3c836937e22a10c19bca0791109efc463118a33fb83c4276` |
| Independent pre-run approval | `f14fd7a37c447317de9f84f1985e5d4afe7c700f2ffe141c00d04fd45de7dda6` |
| Independent post-run verification | `d4b3d84957cbb8e8ac1c4d1b9db660d7a1b6f4a16409dbdb2629735829d210a0` |

Torch2.7.0,Torchvision0.22.0,NumPy1.26.4. The bound runtime inventory includes
Torchvision`_C.so` SHA`8cc7323e6ace3cd92531b2458381b00e6aa8f4f98406601f09e3c301a2952370`,
Torch CPU library SHA`dc3f5537257da1435f82c858a92a4d7316dc6ae973f0e66f698ab39d2a70075c`,
ROIAlign source snapshot SHA`d3a2f0d909d9a7bb47c30cd82058a4afb02453498e3283f62dbf82a64e65a769`
and helper SHA`1e22a97d9b3c84f6c9782fa4db0cb27d03cfd760fa5aa736413df2bc046e724e`.
Full function/file versions and hashes are in preregistration/results. Source
snapshots plus observed parity do not attest a reproducible binary build.

Read-only summary reproduction (does not repeat the benchmark):

```sh
python3 evidence/segmentation_v2/analytic_coordinate_benchmark_002/summarize.py   /home/user/epd-v2-artifacts/analytic-coordinate-contract-002
python3 evidence/segmentation_v2/analytic_coordinate_benchmark_002/independent_reproduce.py
```

Focused regression command:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=tools/segmentation_v2 timeout 120s python3 -m unittest -v   test_analytic_coordinate_r3 test_analytic_coordinate_contract   test_diagnostic_shadow_capture test_training_provenance_audit   test_shadow_mask_paste test_stage_a2_boundary_audit test_stage_a2_boundary_cells
```

## Observations, hypotheses and one next milestone

**Observed:** deterministic original parity, mathematically predicted
origin-dependent displacement, nonadditive paste/alignment effects, and loss
of positive support through coarse sampling/ROI averaging/synthetic bridging.

**Still unknown:** actual trained FPN lattice origin/effective support; learned
head compensation; attribution of real-scene contamination to those mechanisms.
Discretization and nonlinear interpolation remain competing explanations for
residual boundary errors after coordinate consistency. Neither origin was
selected using real/Stage-A2 data, and no production winner is justified.

**One smallest next milestone:** a source-bound, synthetic-only audit of the
actual backbone/FPN lattice coordinates and resize/padding/ROI transformations,
using architectural derivation and impulse probes without trained weights.
Establish the relevant coordinate origin before proposing any production
alignment change. This milestone is recommended, not implemented here.
