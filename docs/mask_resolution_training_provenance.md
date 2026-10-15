# Original 56-grid training supervisor: forensic findings (2026-10-15)

**Decision remains BLOCKED_MISSING_TRAINING_PROVENANCE.** No independently
authentic parent completion or budget-monitor record was recovered. Worker
publication is strongly bound and internally consistent, but does not establish
parent exit0, parent memory sampling or uninterrupted budget enforcement.
The existing #226 admission gate and preregistration remain byte-for-byte
unchanged. No acceptance exception is implemented.

## Why published worker artifacts can coexist with a missing supervisor

The retained `executed_sources/mask_resolution_campaign.py` SHA256 is
`52126dc7eb22dcf7e4bebd94495758c069fcd078dafc152c63e374ea8a7a7aac`.
Its retained `pilot.py` SHA256 is
`3966c9d84ae23a13a3c52582a7091eebbd5da6122444cda830522f5398893692`.
Both match original bindings and current source. The actual lifecycle is:

1. Parent creates exclusive output, admission, source snapshots and binding.
2. Parent `bounded()` opens exclusive worker stdout/stderr files and launches
   the worker with `start_new_session=True`. It samples the child PID's VmRSS
   after quarter-second wait timeouts and kills the child's process group on
   wall/RSS violation. It does not persist launch/PID/sample records.
3. Worker writes configuration, performs each recorded update, and appends,
   flushes and fsyncs its history. After300 records it checks order/frozen state,
   runs validation, saves checkpoint to an exclusive partial file, fsyncs it,
   hard-links it to `epoch_01.pt`, unlinks partial and fsyncs the directory.
4. Worker exclusively writes `result.json`, prints/flushed final worker summary,
   then returns. Worker stdout is not parent stdout.
5. Only **after `bounded()` returns** does the parent call exclusive
   `write_json(process.json, outcome)` and print its own summary. There is no
   `finally` around supervision, pre-launch durable parent record, or parent-death
   kill mechanism for the child. The generic JSON writer uses exclusive create
   but does not fsync file/directory or publish atomically.

Therefore parent interruption (e.g. signal/terminal loss) can leave a worker in
its separate session continuing without the original monitor. A later parent
failure between wait and process publication, write failure, or later removal
of a process file are also compatible with the evidence. No retained trace
chooses among these. Direct worker invocation is another theoretical route,
but shell history contains `run` launch commands, not a recorded direct worker
launch. History commands have no outcomes/timestamps and cannot prove which
invocation ran. Multiple typed launch commands do not prove repeated training;
the exclusive output gate rejects reruns into an existing campaign.

A regression used the **unchanged real bounded runner** with a tiny dummy file
writer, terminated only the tracked dummy parent, and observed the detached
dummy worker publish afterward with no process record. This proves the lifecycle
possibility, **not this campaign's actual interruption cause**. No training,
model construction or inference occurs in that reproduction.

## Verified artifact chain

`training_provenance_audit.py` reads/deserializes tensors with
`torch.load(map_location='cpu', weights_only=True)`; it never constructs a network
or calls forward. It compares every saved tensor/buffer against the pinned
initial checkpoint, verifies original source/admission/configuration bindings,
checks frozen training membership/order using manifest metadata only, validates
history loss arithmetic and hashes retained validation archives without decoding
images. Campaign file SHA256, size, mtime, ctime, inode, device and ownership
inventories match before/after forensics (atime is intentionally excluded).

| Artifact | SHA256 |
|---|---|
| Initial checkpoint | `f3e7531c25cb20920c04283d205d7241028d7321b1ea34e2f1848415744a3b56` |
| Selected checkpoint,186734911 bytes | `bf3ac58116a3a692806ed20b25fb5414277e76da605cf0f60e2e5a23656bfbfe` |
| Frozen dataset manifest | `04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11` |
| Admission | `6720ae5bc0a1a6c94f9640b5f5c584c5ffaa4cf52155f33ffd0719f07065d24a` |
| Binding | `bedd3d0c7589b1ac9a85c8b1aadd200d38e958dd89a261ee38f4626388811530` |
| Configuration | `6822b1c26533fa26f912ed65c93fba33301949b8f8b9114e4116b580eeb1b0eb` |
| History | `2c10eeb6ef97d0bd34bf9863e759af7d3aed4d9168ed74266b2e4b1afe4deb86` |
| Validation metrics | `4f474c62c48f61e4675e6fb62f97295a40cd17fa470ff40fd85d5e47ec6c1c17` |
| Result | `ecc693d2eb5222a63dbeea4c3270d14b9b0e8492b3c80e5266fe2c6feacab133` |
| Worker stdout | `bf2a6ab70f0952084d6594bce426f3f3c66ba3ed0e7c50e3ab4762a10b5733e9` |
| Empty stderr | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Result/binding/checkpoint pins are corroborated by prior #225/#226 evidence;
history and validation hashes also match metadata embedded in the SHA-pinned
checkpoint. Admission/configuration and timestamps are retained artifacts,
corroborated internally rather than separately signed attestations.

Observed checks:

- **12/12 source snapshots** match binding. Configuration embeds exactly that
  binding/policy and enumerates the12 mask trainable tensors.
- **307 checkpoint tensors:** all **295 non-mask tensors/buffers equal** the
  initial checkpoint exactly, and **12/12 mask tensors differ**. This proves
  saved-state equality, not that intermediate non-mask state never changed.
- **300/300 ordered steps**, epoch1,300 unique permitted train scene IDs in the
  exact seeded shuffle; all recorded losses/timings finite and nonnegative,
  weighted mask-loss arithmetic consistent. Records are evidence of worker
  publication under the bound source, not independent per-update tracing.
- Checkpoint embeds exact history/validation/binding/policy, step300/epoch1,
  seed260110 and56-grid architecture. Optimizer has12 parameter states and
  recorded SGD group parameters matching policy. This does not attest each
  historic optimizer transition.
- **12/12 validation archive SHA checks pass**; result equals retained metrics:
 18/22 recovered, IoU0.739693, spill1238, one false positive; recorded eligibility
  rechecks against frozen gate. No new accuracy evaluation or inference.
- Six worker progress summaries reach300; final stdout matches worker result.
  Zero stderr bytes proves only an empty retained stderr file.

## Timing and resource evidence

All dates below are **Europe/London (BST,+01:00)**. Filesystem birth/mtime
receipts and metadata were retained separately; they are not a supervisor clock.

| Field | Actual retained value | Interpretation |
|---|---:|---|
| Admission mtime | 2026-10-13 20:48:55.157757917 | Parent preparation evidence |
| Configuration mtime | 2026-10-13 20:50:34.753899346 | Worker setup publication |
| Result mtime | 2026-10-13 21:23:16.049779718 | Worker result publication |
| Admission→configuration | 99.596141429s | Setup outside worker training timer |
| Configuration→result | 1961.295880372s | Filesystem corroboration only |
| Worker internal training/validation time | 1961.2920848940003s | Monotonic worker claim |
| Summed update timings | 1915.238561623997s | Individual recorded updates |
| Summed validation forward timings | 40.77313324900069s | Retained validation records |
| Remaining internal overhead | 5.280390021002631s | Arithmetic difference; not uniquely attributed |
| Admission→result | 2060.892021801s | About34m21s; excludes parent completion |
| Worker-reported process peak RSS | 3921412KiB | SELF getrusage at result publication |
| Configured cap | 10800s /6144MiB | Configuration, not enforcement proof |

Worker time/RSS fall below configured limits, but neither is an independently
observed parent measurement. Worker timing excludes setup; SELF high-water is
not a process-group/cgroup measurement. No historic parent exit code, sampled
VmRSS peak, wait duration, resource_limit_reason or final stdout was recovered.
Those fields remain **null/unknown**, not exit0 or `resource_limit_reason=null`
as a purported successful original supervision record.

The admission's feasibility process records are genuine and hash-matched:
failed probe exit1/17.902557s, continuation exit0/44.127136s. Their peaks are
2062980/1716808KiB. They describe different workers, **not this campaign's
parent**, and cannot be borrowed as its supervision. The later Stage-A2 exit2
record is also a different process despite referencing this campaign path.

## Independent-source search and provenance matrix

Read-only search examined **19 non-dataset/non-adapter retained process JSONs**;
**zero exact campaign parent command/output matches**. Other process records
were not attributed merely because they mention the campaign.

Available shell history records launches, monitoring, missing-process checks
and tmux commands; it has no exit outputs or per-command timestamps. Its order
can reflect multiple shells, so it does not establish original launch order.
The October13 Codex session file contains no campaign match; later conversation
references are not independent original execution evidence. Journal/syslog and
kernel-log searches for the20:40..21:39 BST campaign window found no campaign-
bound Python completion, OOM-kill or sampled-budget record. Absence of a log
message does not prove absence of a kill/resource violation. GNOME terminal
scope messages cannot be tied to the original parent without retained PID/
start-time identity. OS accounting shows a tmux terminal20:49..21:58 and a host
rebootOctober14; it does not show the training parent's wait status. No live
tmux socket or process-accounting/audit-log directory was available. No deleted
file recovery, other checkout access or external backup access was attempted.

| Provenance item | Status | Scope |
|---|---|---|
| Worker published checkpoint/history/validation/result chain | **VERIFIED** | Artifact publication, not exit status |
| Saved frozen non-mask tensors/buffers | **VERIFIED** |295 exact saved-state equalities |
| Original source/input/config bindings and step ordering | **VERIFIED** | Hash-bound retained chain |
| Internal time, SELF peak, filesystem timeline | **PARTIALLY VERIFIED** | Worker measurements and metadata; no independent parent |
| Shell/tmux launch context | **PARTIALLY VERIFIED** | Command/accounting presence, no completion identity |
| Original parent `process.json` and final stdout | **MISSING** | No authentic completion record found |
| Parent exit, wall, sampled RSS peak, limit reason | **MISSING** | Unknown; never filled from worker claims |
| Continuous wall/RSS budget enforcement | **MISSING** | Configured monitor may have ceased running |
| Historical live `/proc` samples and wait status | **UNRECOVERABLE** | From retained files/current host after reboot; external backups remain unknown |
| Exact cause of missing record | **MISSING** | Compatible mechanisms, none selected |

## Execution, tests and retained outputs

New exclusive output root:
`/home/user/epd-v2-artifacts/mask-resolution-provenance-audit-001`.
The forensic audit completed exit0 (audit success **does not mean admission**),
retaining decision **BLOCKED_MISSING_TRAINING_PROVENANCE**.
Wall16.26s, internal15.878391767s, peak939876KiB; two pinned checkpoints were
**tensor-deserialized**, no model constructed and zero training/inference calls.
Campaign hashes and data metadata were unchanged. No held-out scene files opened.

- `artifact_audit/provenance_report.json`: SHA256
  `ca823b8a69fa99eaa82dd74f0df7a9160050c8fc93426c47d32e59d4c687c13e`.
- `external_search_receipt.json`: SHA256
  `8f1a7e97554185721c93159664c87377a601fe6010220f46d4df7077c1b40f66`.
- `audit_stdout.log`, `audit_stderr.log`, `focused_tests.log` preserve measured
  forensic/test outputs. These are new audit evidence, never substituted for
  the missing original training `process.json`.
- Executed forensic source SHA256:
  `2afb6c8c66567f5579b52b93f6c3988cff09d34758a79d0336265e21fbbde906`.

Reproducible commands (the actual output already exists; do not overwrite):

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 180s \
  /usr/bin/time -f 'forensic_wall=%e seconds forensic_peak_rss=%M KiB' \
  python3 tools/segmentation_v2/training_provenance_audit.py \
  --output <NEW-NONEXISTENT-OUTPUT-DIRECTORY>

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONPATH=tools/segmentation_v2 timeout 120s /usr/bin/time \
  -f 'wall=%e seconds peak_rss=%M KiB' python3 -m unittest -v \
  test_training_provenance_audit test_shadow_mask_paste \
  test_stage_a2_boundary_audit test_stage_a2_boundary_cells
```

**42/42 PASS,0 failures/errors**, unittest0.787s, whole process4.63s,
peak619068KiB. Seven new regressions cover sequence/loss corruption, frozen
buffers, embedded bindings, unknown supervision fields, unrelated process
records, hash/output failure handling and real dummy-parent lifecycle loss.
Focused independent review reported no actionable findings. Broad model/
training/inference suites were not run. No existing runner/gate was changed.

## Two explicit decisions requiring separate authorisation

| Option | Time/cost estimate | Risk and limits |
|---|---|---|
| **A. Qualify the existing hash-bound artifact for diagnostic reuse only through an explicit provenance disposition** | Planning estimate1–2h engineering/review, then previously bounded4-scene shadow capture<=5min/6GiB; no training cost | Defensible for studying a fixed tensor artifact with confirmed history/state, but original budget compliance remains unknowable. Requires independently reviewed, source/checkpoint/purpose-bound authorisation and an explicit new gate; never label original supervision successful or promote/grasp-qualify the model. Not implemented or authorised here. |
| **B. Newly authorise one bounded training run after correcting supervisor retention** | Planning estimate0.5–1 engineering day for runner hardening/tests; on this hardware prior setup+training was about34m21s, prospective cap3h CPU/6GiB, plus new output/checkpoint storage and disk reserve | Additional training compute produces a new candidate, does not repair original provenance, and may differ numerically. Requires durable pre-launch PID/start/binding record, kernel/service-backed budget/lifetime control, crash-safe completion publication and failure tests; no automatic retry. Not implemented or authorised here. |

**Recommended next decision: explicitly authorise Option A solely for the
four-scene shadow diagnostic, with independent review of the source/checkpoint/
purpose-bound provenance disposition.** It addresses the current paste question
without repeating training; it cannot certify missing original supervision.
If original full supervision is a non-waivable requirement even for diagnostic
reuse, choose separately authorised Option B instead. Until a decision is made,
#226 remains blocked, Stage-A2 remains0/2 and no promotion occurs.

### Subsequent explicit Option1 authorisation (2026-10-15)

The user explicitly authorised diagnostic-only qualification after these forensic
findings. [Separate admission and measured experiment](diagnostic_shadow_capture_results.md)
completed under independent review. This changes neither this forensic matrix
nor original supervision UNKNOWN, and leaves the old strict gate blocked.
