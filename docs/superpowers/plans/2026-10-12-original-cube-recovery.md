# Original cube checkpoint recovery

User authorizes safe inspection, original-model reconstruction and one bounded CPU parity check; no training or export.

1. Hash original checkpoint, inspect CPU weights_only=True; reject unsafe/schema/class/tensor drift. Record self-reported metadata separately from authenticated provenance.
2. Reconstruct two-class torchvision ResNet50-FPN/FrozenBN with no downloads. Strict-load every tensor. Compare all original ONNX initializers, including folded FrozenBN constants, and graph preprocessing/postprocessing settings.
3. Test safe loading, wrong hash/schema, class/head mismatch and parity tolerances. Use retained validation scene025 RGB input only, never IDs/held-out/regression scenes.
4. One CPU inference per model,180s/6GiB/2threads,5GiB disk reserve. Predeclare boxes1e-3, scores1e-5, probabilities1e-4 absolute tolerance; labels/count exact and report all >.5 decision differences. Retain outputs/failures, no correction or retries.
5. Document provenance gaps and exact future fine-tuning milestone. Verify original bytes unchanged, review, commit/push draftPR219. No EMD/production changes.
