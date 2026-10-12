# Training dataset expansion

User supplied execution scope: add276 genuine scenes, no training/inference, preserve
pilot/checkpoints/evaluation membership, draftPR219, baseline34b78f6.

1. Verify pinned pilot manifests,24 training sample files, all48 memberships from
   manifests only, capture executable/header/source hashes and dependency/resources.
   Do not open validation/test pixels or predictions.
2. Test and freeze expansion-only scene plan:276 unique IDs/seeds,46 negatives,
   one/two/three targets, separated/touching/partial-overlap arrangements, low/high
   contrast opaque colours, varied orientation/scale/position/light and class0 clutter.
   Same fixed calibrated camera and shader/native raster contract. Add matching static
   collision boxes in new worlds and verify ECM collision/link/visual/model ownership.
3. Capture exactly six batches46, exclusive root and per-scene folders,30s per scene,
   600s per batch,4000s total,2048MiB immediate-process RSS,5GiB free-disk reserve.
   Persist append-only immutable progress records after each attempt and batch seals.
   Stop on any failure; never retry a successful or failed attempted scene. Explicit
   continuation may verify immutable completed batches and list unattempted scenes.
4. Validate every raw sample, native mappings/labels,RGB uniqueness and scene hashes
   against entire pilot manifest/historical hash exclusions. Report authored/visible
   counts,areas,bbox overlap, estimated geometric occlusion,colour/light distributions.
   Preserve valid hard scenes including fully occluded objects within positive scenes.
5. Compose hash-bound300/12/12 dataset manifest and backwards-compatible loader.
   Enumeration reads metadata only; training iteration loads exactly300 training files
   and existing train.tensors labels. Supervised loader smoke has no model/optimizer.
6. Focused tests cover freezing,duplicates,labels,split isolation,partial failures,
   membership and legacy loading. Verify actual300-sample loader, all retained hashes,
   snapshot evidence, commit/push/update draftPR. STOP no training/export/evaluation.
