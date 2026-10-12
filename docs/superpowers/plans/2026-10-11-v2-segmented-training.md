# Deterministic segmented CPU campaign

User explicitly authorizes implementation and Stage1 only, baseline b6dbacc.
Preserve historical failed preflight and all recipe/gate/split defaults.

1. Test deterministic actual PyTorch fixture with stochastic forward, SGD momentum,
   Python/NumPy/Torch RNG and deterministic epoch ordering, continuous8 vs4+4.
   Negative tests cover altered checkpoint/config/dataset/source/history/epoch.
2. Implement atomic checkpoint save and complete-stage hash seals, verify before
   loading, restore every state after model construction. Save segmented checkpoint
   after validation so its RNG is exactly the next-epoch boundary.
3. Version two4-epoch stages,10800s each,18000s aggregate,6144MiB,5GiB diskreserve,
   300samples/epoch. Measure stage-specific feasibility with unchanged1.5 factor
   and120s overhead per stage; include48 validation forwards/stage. STOP iffails.
4. Execute only Stage1 epochs1–4,1200steps, everyepoch validation/checkpoints/
   optimizer/RNG/losses. Bounded process, immutable evidence, no retries.
5. Verify complete epoch4 checkpoint and history; project Stage2 from measured
   training+validation runtime and remaining aggregate allowance. Do not execute it.
6. Review, focused regressions, commit/push draftPR219. No testpixels/historical
   regression/model export/motion/EMD modifications. Report only actual evidence.
