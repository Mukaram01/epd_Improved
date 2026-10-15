# R4 execution and engineering decision

## Verified single execution

The user separately authorised exactly one execution of the already reviewed
proposal. It ran from commit`5b6fff3343b82fb50be635004e7c9e16901ea8df`, on the
existing draft#228 branch. Source, preregistration and review were unchanged.
No retry, overwrite, new model experiment, real-scene access or learned-model
forward occurred. Earlier failed001/completed002 evidence remains intact.

**COMPLETE_SYNTHETIC_ONLY:**219/219exact case identities,657/657scalar graphs,
876/876processing chains,4/4finite correctly shaped trace archives. All six
patterns, three kernels, small odd/even rasters, fractional phases and three
full800anchors are present. Actual construction was inspected on meta; all
numerical graph parameters were artificial, as preregistered.

**79/79focused tests PASS**,0failures/errors; unittest1.501s,process4.75s,
peak632464KiB. The full benchmark is separate from those tests. No benchmark
source or production code was modified in this task.

Supervisor exit0,wall**20.644817491s**,peak worker RSS**658396KiB**; numerical
worker timer**15.032926627s**. CPU-only,2threads,300s wall,2GiB worker RSS.
Resource-limit reason null, parent-death SIGKILL and process-group cleanup
recorded; stdout/stderr both empty. RSS uses20ms sampling plus wait4 high-water
rejection, not kernel memory.max. These are measured supervisor/worker times;
no unmeasured parent startup duration is claimed.

## Numerical checks and measurements

| Comparison | Checked count / result | Maximum absolute difference |
|---|---|---:|
| Scalar backbone stage outputs, NumPy vs Torch |4levels×657graphs |2.384185791015625e-7 |
| Actual controlled FPN vs independent NumPy |4levels×657graphs |4.76837158203125e-7 |
| Controlled mask-head convolutions |876chains,4convs each |4.76837158203125e-7 |
| Independent half-pixel resize vs actual preprocessing |5input sizes |4.76837158203125e-7 |
| OpenCV uint8 upsample vs independent continuous bilinear |256→512 ramp |0.25LSB |
| Nearest top-down index sampling |3links×657graphs,bitwise |0 |
| ROIAlign / unit deconv / original paste / binary union |876receipts for each,bitwise |0 |
| Box transformed→native512 resize |876chains,bitwise |0 |
| Actual MultiScaleRoIAlign selection |3anchors×4boxes,bitwise |0 |

Float differences are below the preregistered3e-6 tolerance; uint8 difference
is below1LSB. Bitwise operations used zero tolerance. Independent review
verified source-bound parity gates and receipts, not a second657graph execution.
Four retained traces additionally reproduce P energy/nonzero/centroids and
strict native ROI/union counts directly from arrays. Full graph operands for
all cases were not retained, so independent replay of every intermediate would
require rerunning; that was neither done nor permitted.

Preprocessing resized/padded sizes:

| Input H×W | Resized | Padded |
|---|---|---|
|256×256 |800×800 |800×800 |
|255×257 |800×806 |800×832 |
|257×255 |806×800 |832×800 |
|511×513 |800×803 |800×832 |
|512×512 after synthetic uint8 upsample |800×800 |800×800 |

At full800, the32-pixel translation gives **exact zero boundary-free residual
in all12kernel×level combinations**. Valid point counts for P2/P3/P4/P5 are
6561/1521/289/49 per kernel. A1-pixel shift is not an integer shift on these
feature grids; maximum boundary-free value residuals are:

| Level / stride |1px shift maximum, all three kernels |32px shift maximum |
|---|---:|---:|
|P2 /4 |0.00499415397644043 |0 |
|P3 /8 |0.003745555877685547 |0 |
|P4 /16 |0.002496957778930664 |0 |
|P5 /32 |0.001248478889465332 |0 |

These are **feature-value differences, not pixel displacement measurements**.
Full-frame32px residuals remain nonzero at artificial-kernel boundaries:
symmetric P2–P5 mean absolute errors .054009/.052422/.046206/.027158;
right-tap .117778/.111040/.082308/.042787. Boundary-free errors remain zero.
Small deep rasters have no valid structural interior and report null; they
cannot establish interior equivariance. The full800anchors cover x-ramps only.

Sampling depends on phase and kernel. Among12small impulse cases per kernel,
zero P2/P3/P4/P5 energies occur2/2/2/5times for centre and right-tap kernels,
and0/0/0/0for symmetric kernels. These are finite-raster, conditional artificial
responses, not proof of irrecoverable information or a trained response origin.
All36empty cases have zero feature energy at every level. Fractional-phase
checkerboards exhibit their declared interpolation/periodic symmetries; border
and empty phase repeats are controls rather than independent replicates.

Across876artificial mask chains, native512foreground totals**15,887,023**,
mapped256foreground**4,009,545**, inactive native slots selected by union
**151,157**; exactly`4*mapped-native=inactive`. Counts repeat synthetic cases;
there is no renderer truth, IoU, real background-spill or safety interpretation
for these arbitrary amplitudes/logit biases. They verify the conservative
mapping contract, not perception quality.

## Independent verification and retained evidence

`/root/r4_verification`, distinct from the implementer, independently parsed all
raw results, rehashed36source bindings, verified completeness/parity receipts,
5,256scalar chain invariants, finite values, traces, logs and supervision.
They independently reproduced the numerical summary, then matched291fields
of the implementer's offline summary. All93old manifest artifacts (9failed001,
84completed002) still match. No actionable execution findings remained.
Decision:**VERIFIED_COMPLETE_SYNTHETIC_ONLY**.

Output:`/home/user/epd-v2-artifacts/source-lattice-contract-001`.
Raw results and all four traces remain in this exclusive directory. Committed
receipts, manifests, independent scripts/reviews and summaries are in
`evidence/segmentation_v2/source_lattice_r4_001`. No full output is replaced.

| Evidence | SHA256 |
|---|---|
| Executed source |`4398e1a9dd3754d7bd31cfefff01c6db53907833034bc2bf6a48a94112d4f4fe` |
| Original preregistration |`f21e4277b74908228003b58ba3134de459b28948b6eeeb542b7b5dd3eb3870c1` |
| Original methodology review |`e27ba46e7ce4622ff6bddd34464453334be08b446fd2cc2056cecd9d8b1f5af8` |
| Raw results |`e4fd0da052a0a9051dbbfff584731375a24e4fb0060ba3be433153008ed84e7b` |
| Supervisor receipt |`aaf7f377741f10411a294580a734307fa0b65fdef4f53ae88012edb0648e4076` |
| Offline summary / full output manifest |`7fb62622c203ec1a116b8d96ba8078a5f982ed6891291ec9fe6cc511cc8b8bdb` |
| Independent verification |`afa46bf1cacd15d2afb8c72a9fb37a321e0581f31c8716ad17fbed9c418a5fab` |
| Independent summary comparison |`bfe3d3920898b2539ac87d3b51c2519ddbee323ed1aea565599ca56714e6f066` |

Exact command executed once:

```sh
python3 tools/segmentation_v2/source_lattice.py \
  --policy tools/segmentation_v2/source_lattice_preregistration.json \
  --review evidence/segmentation_v2/source_lattice_proposed_001/independent_methodology_review.json \
  --review-sha256 e27ba46e7ce4622ff6bddd34464453334be08b446fd2cc2056cecd9d8b1f5af8 \
  --authorise-full-run
```

The original command now rejects an existing output; **do not rerun it**.
Read-only summary reproduction:

```sh
python3 evidence/segmentation_v2/source_lattice_r4_001/summarize.py \
  /home/user/epd-v2-artifacts/source-lattice-contract-001
python3 evidence/segmentation_v2/source_lattice_r4_001/independent_verify.py
```

## Engineering decision — one corrective implementation candidate

1. **Conclusive:** source geometry and declared operators are reproducible;
   FPN phase/boundary effects and coarse sampling are real conditional effects.
   R3/R4 do not identify a universal corrective shift. Earlier #226 real frozen
   logits showed all tested paste controls fail no-harm: lower spill traded for
   missing foreground or cross-object harm. No production geometry change has
   earned selection.

2. **Material uncertainty:** effective learned logits/support and disagreement
   between independently rendered512/256labels prevent attributing real spill
   to a single correctable coordinate offset. Original training supervision is
   also UNKNOWN and independently blocks production qualification. More
   synthetic coordinate audits will not resolve these issues.

3. **One implementation milestone:** implement a **training-only native512
   post-paste boundary auxiliary loss for the mask head**, retaining original
   production paste, boxes, confidence, strict>.5 threshold and ANY2×2 mapping.
   Standard ROI56mask loss remains; the auxiliary directly supervises the
   differentiably pasted native512probabilities against existing admitted
   native512training labels, with balanced FP/FN terms in a fixed2pixel band
   on both sides of the target boundary. No inference trimming, erosion or
   oracle labels. All non-mask tensors/buffers remain frozen. The present task
   implements or trains **none** of this.

   Why this candidate: #226 baseline native-ROI spill has427/441pixels within
   2pixels of foreground. Existing`train.tensors` already supplies native512
   targets, but installed`maskrcnn_loss` projects them to ROI-grid targets before
   loss; the proposed post-paste consumer-space objective is materially different
   from simply repeating epochs or increasing ROI-grid boundary weights.
   This is an evidence-backed **hypothesis**, not demonstrated causation or an
   expected performance gain. Native512-only supervision cannot directly
   penalise all512/256raster disagreement cells and may fail strict acceptance.

4. **Exact prospective pass/fail:** after separate implementation/training/
   inference authorisation, predeclare one fixed loss configuration and a
   provenance-complete bounded run; no parameter sweep or acceptance-scene
   selection. Use the full permitted validation split and frozen baseline
   assignments/thresholds, including negative and small/thin instances.
   Require≥20%aggregate native-ROI **and** mapped-background spill reduction,
   **no per-scene/per-instance increase** in missing foreground or cross-object
   contamination, no per-instance IoU decrease, no dropped baseline detection
   and zero confident false positives on negative scenes. On the previously
   reported four-scene sentinels, the numerical gates are native-ROI spill
   **≤352 vs441**, mapped spill**≤171 vs214**, native missing**≤1131**,
   mapped missing**≤214**, cross-object**≤285/92**, all with the stricter
   per-scene/per-instance no-harm rules. The20%effect size is a prospective
   decision threshold, not a measured or predicted gain. These criteria must
   also hold across the full validation split, not only four sentinels.

   Improved validation is a development result, not promotion. Production
   requires a separately authorised independent qualification and genuine
   strict Stage-A2 **2/2**, unchanged association rules, without tuning on that
   capture. Full genuine initializer/training/supervisor provenance is mandatory;
   using an UNKNOWN-provenance initializer cannot silently become production
   qualified. The existing unmatched instance cannot be recovered by a frozen
   proposal/box head, so this candidate targets mask quality, not proposal recall.

5. **Invalidation:** reject the candidate if spill reduction is bought by lost
   foreground, thin-object/negative-scene harm, cross-object increase, failed
   validation materiality, changed frozen state or absent provenance. If it
   improves native masks but leaves strict mapped association at0/2, it has not
   solved the deployment problem and remains unpromoted. Resolution-disagreement
   labels or learned-support effects may dominate; do not relabel those failures
   as success or respond with another synthetic-only coordinate audit.

Stage-A2 remains **0/2 BLOCKED** and original training supervision **UNKNOWN**.
No new correction, retraining, real-scene evaluation, safety-gate relaxation,
EMD change, ROS/Gazebo or motion was performed. The next decision is whether to
separately authorise implementation of this single training-only candidate.
