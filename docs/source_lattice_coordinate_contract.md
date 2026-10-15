# Source-defined EPD coordinate ledger and proposed synthetic benchmark

## Disposition

**PROPOSED / NOT EXECUTED.** This branch stacks on draft#227 at
`e0851e081e74192b0929d71d2956830e202b4cbb`. #227 completed1944cases/7776arms/72traces
with bitwise original parity. Its001failed and002completed output manifests
were verified unchanged. No historic experiment is rerun. Stage-A2 remains
**BLOCKED0/2**; original training supervision **UNKNOWN**.

The task authorises source audit, implementation, tests, independent methodology
review and a draft PR. It explicitly excludes execution of the full new matrix.
No checkpoint, trained parameter, real scene or renderer label is read. The
actual EPD model is constructed on the **meta device for structure inspection
only**, with no forward or parameter values. Synthetic operators below use
fully specified artificial kernels and cannot identify trained response centres.

## Actual construction and pinned identity

EPD `recover_original.make_model` constructs `resnet_fpn_backbone(resnet50,
weights=None,norm_layer=FrozenBatchNorm2d,trainable_layers=3)`, then `MaskRCNN`
with2classes, min_size800, max_size1333, mean[.485,.456,.406],
std[.229,.224,.225], explicit box/RPN confidence/NMS settings. EPD resets every
FrozenBN epsilon to0. `mask_resolution.configure56` changes only mask ROI pool
14→28, requires sampling_ratio2, and restricts trainability to mask head/predictor.
The classifier's actual head is four256-channel3×3,s1,p1 convolutions/ReLUs,
then256-channel2×2,s2,p0 transposed convolution/ReLU and2-channel1×1 logits.
No alternate MaskRCNN factory or stock pretrained defaults are assumed.

The installed construction is independently checked on meta, including ResNet
block counts/stride placement, FPN lateral/output shapes, normalization epsilon,
transform and mask pool/head. Runtime versions: Torch2.7.0,Torchvision0.22.0,
NumPy1.26.4,OpenCV4.11.0.86. Full file/binary hashes are in
`evidence/segmentation_v2/source_lattice_proposed_001/source_bindings.json` and
in the preregistration. Original #227 runtime inventory is required to match.
Bindings cover EPD construction, Torchvision ResNet/backbone/FPN/poolers,
transform/paste, Torch convolution/pooling, compiled Torch/Torchvision/OpenCV,
the retained mapping source and C++ RGB01 consumer adapter. Reading retained
executed **source code** does not read its accompanying capture pixels.

## Coordinate ledger

Use image-edge coordinates: pixel index i has centre i+.5, image interval[0,W].
A half-pixel interpolation source **index** is `(j+.5)*Win/Wout-.5`; its edge
coordinate is `(j+.5)*Win/Wout`. Boxes use continuous edge coordinates. Index
coordinates and physical/support coordinates must not be silently interchanged.

| Stage | Actual source-defined operation | Coordinate/support consequence |
|---|---|---|
| Retained RGB input | Python diagnostic RGB256 uint8, OpenCV INTER_LINEAR→512, CHW float32/255 | Native512 centre j+.5 samples RGB256 index(j+.5)/2−.5; uint8 interpolation quantizes. C++ RGB01 accepts exact raster BGR8 and swaps to RGB/255 with **no implicit resize**. |
| Normalization | Per-channel subtract mean/divide std before resizing | No geometric shift; amplitude/sign may affect downstream nonlinear response. |
| Transform resize | min(800/min(H,W),1333/max(H,W)), bilinear align_corners=False, recompute_scale_factor=True | Output dimensions floor(H*s),floor(W*s); actual axis ratios use these integer dimensions, so nonsquare rounding matters.512square→800square; inverse centre map .64*(j+.5) in native512 edge coordinates. |
| Box resize | `resize_boxes`: x multiplied by newW/oldW, y by newH/oldH | No half-pixel additive term for continuous boxes. Postprocess reverses to original512 before paste. |
| Batch padding | Zero right/bottom padding to max batch H/W rounded up to32 | No top/left offset; boundary values change.800square already divisible32. Odd resized dimensions may pad to832, etc. |
| Stem | ResNet50 V1.5 conv7,s2,p3; FrozenBN/ReLU; maxpool3,s2,p1 | Nominal support anchor stays input edge.5; strides2→4; RF7→11. Maxpool chooses data-dependent support within that envelope. |
| C2 / layer1 | 3bottlenecks,3×3 stride1 | Jump4,deepestRF35,anchor.5. |
| C3 / layer2 | 4bottlenecks,first3×3 stride2 | Jump8,deepestRF91,anchor.5. |
| C4 / layer3 | 6bottlenecks,first3×3 stride2 | Jump16,deepestRF267,anchor.5. |
| C5 / layer4 | 3bottlenecks,first3×3 stride2 | Jump32,deepestRF427,anchor.5. |
| FPN | C2–C5 lateral1×1→256, nearest top-down addition, output3×3,s1,p1 | Tensor index labels inherit lateral stride; support includes phase-dependent coarser ancestry. Output3×3 does **not** feed the next top-down stage. |
| Extra pool | LastLevelMaxPool k1,s2 on P5 | P6 stride64,subsamples P5. RPN can use this; mask/box ROI feature names0,1,2,3 exclude pool. |
| Level mapper | floor(4+log2(sqrt(boxarea)/224)+1e-6),clamp2..5 | Uses resized/transformed box coordinates, not native512.32/112/224/448square examples map2/3/4/5. |
| Scale inference | nearest power-of-two ratio of feature shape to max unpadded resized image shapes; first axis used | Actual padding/strides produce1/4,1/8,1/16,1/32; check each, do not substitute padded size as image_shapes. |
| ROIAlign | Actual MultiScaleRoIAlign28,sampling_ratio2, calls roi_align without aligned argument | Default aligned=False. Start b0/stride; extent max((b1−b0)/stride,1); each bin averages2×2 bilinear feature samples; out-of-feature samples clamp/zero per pinned C++ code. |
| Mask head | Four padded3×3 conv/ReLU then2×2,s2 transpose/ReLU,1×1 classifier |28lattice retains nominal stride/anchor through symmetric padding; structural RF9ROI bins after four convs.56output has two nonoverlapping children per axis per parent before channel/class mixing. Effective response is weight/data dependent. |
| Mask probabilities | Class selection and sigmoid, using learned class logits in real model | No spatial transform; synthetic benchmark substitutes explicitly artificial scalar logits. |
| Native paste | Probability pad1:56→58; expand box58/56; cast int64; bilinear align_corners=False resize to inclusive extent(end−start+1); clip into512 | Truncation toward zero differs from floor for negative expanded boxes. Integer phase, inclusive extent and boundary clipping prevent a globally exact affine inverse. |
| Consumer ROI | RGB01 C++ decode clips before integer truncation, crops full-image probability mask to half-open integer ROI; Python audit uses equivalent clipped truncation | Later C++ geometry floor/ceil sees already integer RGB01 boxes. Do not apply legacy-mask resizing path to RGB01 full-image masks. |
|512→256 | Retained `map_mask` applies strict>.5 within integer ROI, embeds512, binaryANY2×2 | Native sample indices2j/2j+1 map to source cell j. Sample centres are j+.25/j+.75 in256 edge units. This is conservative support union, not probability interpolation or a centred downsample. |

### Receptive-field derivation and FPN phases

For an odd convolution/pool: jump'=s*jump, RF'=RF+(k−1)*jump,
anchor'=anchor+((k−1)/2−padding)*jump. Start jump1,RF1,anchor.5.
ResNet50 here places downsampling stride on the3×3, not first1×1: hence
35/91/267/427 rather than a different ResNet variant's RF. Residual shortcuts
have smaller supports; ReLU/FrozenBN can suppress contributions. Deepest-path
RF is structural support, **not a learned centre of mass**.

At800, C2–C5 are200/100/50/25. P-level nearest target resizing is exactly2×;
child i inherits parent floor(i/2). In transformed-image units, that parent's
nominal anchor differs from child's by `−stride_child*(i mod2)`. Repeated
ancestry uses floor(i/2^k). For arbitrary non2× sizes nearest uses
floor(i*Nsource/Ntarget); separate odd-shape operator tests cover this. Actual
batch padding32 makes the normal C2–C5 top-down size ratios exactly2×.

For interior levelL∈{2,3,4,5}, q=2^(5−L), final output3×3 accesses C5 ancestor
indices floor((i−1)/q)..floor((i+1)/q). Its deepest support envelope edges are
`.5+32*left−427/2` and`.5+32*right+427/2`. The widths are:
P2/P3 **427 or459** by phase, P4 **459**, P5 **491**. Lateral supports fit inside
these deepest envelopes here. Frame padding clips actual contributors. This
is a phase-dependent support ledger, not one affine response-centre lattice.
`coordinate_ledger.json` retains each phase's intervals for independent checking.

### Which R3 origin assumptions survive the source audit?

The **fixed-input-centre** origin `.5+j*stride` matches the nominal convolutional
support anchors of C2–C5 in **transformed input coordinates**. The **cell-centred**
origin `(j+.5)*stride` is not that backbone support anchor for stride>1. Neither
assumption identifies the effective learned FPN response after top-down mixing.

If feature samples are interpreted at nominal backbone anchors, legacy ROIAlign
has a+.5 transformed-pixel ramp offset, and aligned=True has `.5−stride/2`.
At P2 this is+.5 versus−1.5 transformed pixels, or+.32 versus−.96 native512
pixels. This is conditional coordinate algebra, **not a measured error of trained
features**. A source-defined tensor coordinate convention does not uniquely
locate semantic evidence learned from data.

Combined resize centre geometry maps nominal800anchor.5 to native512edge.32
and original256edge.16 **before boundary clamping**, with backbone stride
multiplied by.64/.32 respectively. The first interpolation sample is clamped
to the first source pixel; these extrapolated geometric anchors are not the
actual response centre of the boundary sample.
For each transformed input index j, native512 interpolation taps are the
clamped floor/ceil neighbours of `(j+.5)*.64−.5`. Each native tap k in turn
uses clamped RGB256 neighbours of `(k+.5)/2−.5`. The original-image structural
support is the union of these tap sets across the backbone support, with
zero-weight taps excluded. Padding, learned weights and nonlinear switches
can shrink effective support. A single RF width scaled by.32 does not describe
that full discrete resampling support.
The two sequential resize filters and uint8 quantization remain: composition of
centre maps is not numerical equivalence to a single direct resize.

The transposed2×2 stride2 operator maps childu to parentfloor(u/2). Relative to
canonical56bin centres, the parent's28bin centre lies+.5child-bin for evenu
and−.5for oddu. Learned taps/channels can encode different responses; artificial
unit deconvolution cannot establish actual mask-logit response centres.

**UNKNOWN:** learned receptive-field/semantic response centres, trained kernel
asymmetry, residual/top-down relative contribution, maxpool switches on real
images, learned proposal offsets and head compensation. No trained weights or
scenes are used to resolve them.

## Preregistered proposed experiment

The proposed source-lattice benchmark is scalar and deterministic. It preserves
the actual geometric schedule and operators, not256-channel learned semantics.
Actual meta construction guards the schedule. Scalar synthetic bottlenecks use
identity1×1/FrozenBN and .5main+.5shortcut weights, with nonnegative fields so
ReLU is an identity; source residual addition is preserved. These are artificial
weights, not a full ResNet model or a learned head. Actual installed
FeaturePyramidNetwork([1,1,1,1],1) is used with every lateral/output parameter
overwritten by declared kernels; constructor random values are never used.

Fixed spatial kernels: centre delta; separable binomial symmetric3/7tap;
one-pixel right-tap. They deliberately distinguish nominal support anchors from
conditional amplitude centroids. No alternative or parameter is selected from
outcomes. Reference convolution is independent NumPy accumulation; actual
functional convolution and installed FPN are checked against it. Float conv,
FPN and resize maximum absolute tolerance is **3e-6**, fixed before execution.
This is numerical tolerance, not a geometry uncertainty bound. Nearest, direct
ROIAlign, replication/deconvolution, original paste, box resize andANY2×2 must
match bitwise. OpenCV uint8 resize is checked against independent continuous
bilinear geometry to≤1LSB (integer rounding is not claimed bitwise identical).

216small cases: sizes64×64,65×67,95×96 ×6patterns ×4phases ×3kernels.
Patterns: tent impulse, x/y ramps, bilinearly translated infinite checkerboard,
one-pixel frame border, empty. Phases0,.25,.5,.75 shift impulse centre, ramp
sample argument, and checkerboard bilinear phase. Border/empty deliberately
repeat across phases; they are controls, not independent phase replicates.
Three additional **800×800 x-ramp/.25phase anchors**, one per kernel, supply
nonempty boundary-free regions. Total **219cases**, each translated diagonally
by0,1,32pixels before scalar backbone/FPN: **657graph evaluations**.
Small fields are padded right/bottom32; they are geometric/boundary stress
rasters, not a claim that EPD's production transform resizes to64or96.

Every baseline case probes all four P levels through ROI28, four controlled
head convolutions, unit deconvolution56, artificial logit bias−.5, box postprocess
and original paste512, integer ROI andANY256: **876mask chains**. Small-raster
probes use transformed box[8.25,10.75,48.25,50.75] and their own transformed size;
full800anchors use interior[256.25,272.75,480.25,496.75]. Native paste boxes are
independently checked against resize_boxes(transformed_size→512), not reused
unscaled. Per-level probes are forced calibration probes; actual level selection
is separately checked with MultiScaleRoIAlign on all four full800anchor levels
using32/112/224/448square boxes and actual800 image_shapes.

Additional preprocessing checks: actual normalization/resize/padding for
256square,255×257,257×255,511×513 and the real synthetic256uint8→OpenCV512→800
path. No actual image file is read. Four first-case per-level traces retain
input/C/P/paste arrays. All219case identities,657translation records and876mask
receipts must complete; failures retain partial evidence and reject completion.

Metrics: per-level energy/support/energy centroid (null when empty), source
parity maxima, full-frame translation residual, integer lag/remainder, and
conservative structural-boundary-free residual where valid points exist.
For32-pixel shifts, strides divide the shift;1-pixel shifts test phase/aliasing
without assuming equivariance. Small deep layers may have **no boundary-free
points**, reported null. Original paste/union counts and inactive2×2 slots expose
support amplification. There is no analytic scene label or production winner.
Artificial-kernel impulse centroids are observations conditional on those
kernels; they cannot identify a trained feature origin.

## Proposed execution boundary

Fixed exclusive output (must not exist):
`/home/user/epd-v2-artifacts/source-lattice-contract-001`.
CPU-only,2threads,300s wall,2GiB worker RSS, parent-death SIGKILL,
process-group cleanup, durable stdout/stderr/launch/exit/wall/peak/limit reason.
RSS is20ms polling plus wait4 high-water rejection, not kernel memory.max.
Failure/timeout/drift/parity error stops; no automatic retries or relaxed gates.
The implementer will not execute the full matrix in this task.

Separate user authorisation **and** the SHA-bound independent methodology
approval are required before using the command printed in the final review
section. The CLI additionally requires `--authorise-full-run`; this flag is an
operator acknowledgement, not a replacement for genuine human authorisation.
Source/operator identities and meta structure will be rechecked at execution.

## Next decision

After reviewing the source ledger and independent findings, decide whether to
authorise **one** bounded execution of this fixed synthetic proposal. It cannot
qualify the trained model, change production paste or clear Stage-A2/provenance.


## Verification, independent review and proposed command

**79/79 focused tests PASS**,0failures/errors; unittest1.499s,whole process4.81s,
peak641760KiB. Twelve new focused tests cover actual meta construction,
source-defined RF/phase coordinates, odd/even padding/resize, independent
controlled convolutions/FPN, one tiny17×19 scalar schedule,512paste/256union,
strict threshold, drift/exclusivity/preservation, incomplete rejection and the
separate-authorisation gate. Existing supervisor tests exercise real disposable
wall/RSS failures and durable evidence. This test execution is not the proposed
219case/657graph benchmark.

Independent reviewer`/root/lattice_review` rehashed all36 source bindings,
independently passed the12 focused tests, verified the coordinate ledger and
confirmed the new output directory is absent. Earlier findings about512paste,
800→512 box transforms and fractional phase semantics were corrected before
approval. Decision **APPROVE_PROPOSED_METHODOLOGY_ONLY**;
`execution_authorised=false`. The genuine record is retained in
`evidence/segmentation_v2/source_lattice_proposed_001/independent_methodology_review.json`.

Preregistration SHA256:
`f21e4277b74908228003b58ba3134de459b28948b6eeeb542b7b5dd3eb3870c1`.
Independent review SHA256:
`e27ba46e7ce4622ff6bddd34464453334be08b446fd2cc2056cecd9d8b1f5af8`.
The complete bound source/ledger hashes are in the preregistration and evidence
source_bindings.json; no source install/update occurred.

**Do not execute until separately authorised by the user:**

```sh
python3 tools/segmentation_v2/source_lattice.py \
  --policy tools/segmentation_v2/source_lattice_preregistration.json \
  --review evidence/segmentation_v2/source_lattice_proposed_001/independent_methodology_review.json \
  --review-sha256 e27ba46e7ce4622ff6bddd34464453334be08b446fd2cc2056cecd9d8b1f5af8 \
  --authorise-full-run
```

Focused regression reproduction (does not execute the full benchmark):

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONPATH=tools/segmentation_v2 timeout 120s python3 -m unittest -v \
  test_source_lattice test_analytic_coordinate_r3 test_analytic_coordinate_contract \
  test_diagnostic_shadow_capture test_training_provenance_audit \
  test_shadow_mask_paste test_stage_a2_boundary_audit test_stage_a2_boundary_cells
```

No full results, wall/peak benchmark measurements or synthetic full-matrix
findings are claimed. The next decision is separate user authorisation for
exactly one reviewed bounded synthetic run. Stage-A2 and provenance remain
unchanged, regardless of whether that future diagnostic completes.
