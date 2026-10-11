# Bounded EPD v2 training campaign

Starting identity: 0e83096cd239c352e5cd384eb5c46e66945da5fd, draft PR219.
User supplied the full implementation and acceptance scope; this plan executes it.

1. Verify manifest/plan identities and all train/validation files with existing
   verified_rows. Verify all 48 recorded acquisition outcomes and frozen memberships
   from manifests; do not open held-out scene files. Hash-check initialization,
   prior smoke (identity only, never load it for initialization), configs and sources.
2. Record current CUDA/dependency/memory/disk state and compare CI job logs at
   the starting head and base. Fix only a demonstrated introduced regression.
3. Freeze campaign configuration and the exact user gate before measurement:
   recovered>=18/22, mean IoU>0.685890, false positives<=1, background spill<1915,
   zero class/identity/output contract violations. Confidence>=0.8, masks>0.5.
   Eligible ordering: recovered, mean IoU, lower spill, earlier epoch.
4. CPU feasibility only: fresh pinned COCO model, deterministic forward/backward
   on representative train images WITHOUT optimizer updates or checkpoint saving;
   validation forward timing only. Declare 3600 s wall/6144 MiB RSS for a complete
   CPU campaign; CUDA retains1800 s. Estimate all192 train steps and96 validation
   forwards using measured worst-case times, 1.5 safety factor plus120 s overhead.
   STOP if estimate or memory projection exceeds budget. No second smoke model.
5. If feasible, add/test a full-campaign runner using existing build_model/tensors/
   verified_rows, preserving model recipe. Exactly one campaign, batch1, seed260110,
   <=8 epochs, fresh pinned COCO, save every epoch+optimizer/history/curves externally.
   Validate each completed epoch only on frozen12 validation scenes, retain all
   raw predictions and per-object/detection/negative results and choose by frozen gate.
6. If no eligible checkpoint, retain evidence, report rejection and STOP. If eligible,
   export fixed RGB01 images[1,3,512,512], dynamic N boxes/labels/scores/full-image masks,
   run structural/CPU/raw-PyTorch parity and actual C++ checks on validation inputs.
   Preserve every failure, no mask corrections, no held-out/historical regression.
7. Push implementation/tests/compact evidence to draftPR219; confirm draft. One final
   status from the user's allowed list; never merge, deploy, move robots or alter EMD.
