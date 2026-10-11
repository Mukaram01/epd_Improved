# Adapter verification, 2026-10-11

Classification: **ADAPTER_READY_FOR_V2_EXPORT**.

- Full C++ application, standalone production-path probe and focused tests built
  using the existing ROS Humble/vendor/OpenCV installation.
- 10 adapter C++ cases and the observation, scheduler and geometry regression
  executables pass. See XML for individual results, and `ctest.txt`.
- 24 retained, hash-verified validation prediction sets, 58 confident instances:
  zero differences in boxes, labels, scores and ROI mask bytes. These execute
  constant-output test graphs through real OrtBase/P3OrtBase, not a model-quality
  evaluation. Dynamic N=0/N=1 and eight malformed graphs also pass their checks.
- Localization/tracking consume the exact cropped masks and strict >0.5 threshold.
- Legacy preprocessing bytes, mean/order/layout/padding, > confidence boundary,
  ROI mask semantics and float box-scaling edge are covered in actual C++ tests.
- Historical SHA-pinned ONNX model: one genuine CPU inference on validation
  scene_025, 3.373738442 seconds, zero differences against retained reference.
  This used the historical RGB256->RGB512 float32 /255 contract, not legacy BGR
  mean preprocessing. It does not validate an unexported trained v2 model.

The historical executed binary is preserved outside Git. The final binary adds
legacy-only rounding preservation and probe consumer checks; hashes and locations
are recorded separately. Existing external datasets/checkpoints and their prior
hashes remain intact. No held-out input or renderer ID tensor was read by a model.

During implementation the compiler caught a removed tracking-map call, which was
restored. A malformed-shape test initially reused the valid score shape and was
corrected. Review caught a legacy float-versus-double pixel-edge regression;
`legacy_rounding_before_fix.txt` retains its failing test, now fixed and covered.
Final builds contain the pre-existing unused `ortStatusMessage` warning in CPU
builds; no compilation errors remain. Repository-wide model-dependent tests are
not claimed: their historical bundled models are unavailable in this checkout.

No adapter blocker remains. Supply an accepted v2 checkpoint, export the documented
interface and run pinned raw PyTorch/ONNX/C++ comparisons. Model quality,
conservative association and physical-contact qualification remain unproven.
