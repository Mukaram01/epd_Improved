# EPD Training Studio

EPD-4 adds a Training Studio to the existing Train workflow. It does not replace the current P2/P3 dockerized training/export backend; it makes that backend observable and recoverable.

## Open Training Studio

Open **Train Vision Model**, then select **Training Studio** in the Train header.

The normal preparation flow is unchanged:

1. Choose P2 (Faster R-CNN) or P3 (Mask R-CNN).
2. Choose the model architecture.
3. Choose the label list.
4. Choose or generate a `custom_dataset` with `train_dataset/` and `val_dataset/`.
5. Validate training readiness.
6. Open Training Studio to inspect dataset statistics and checkpoint state.
7. Start Train.

## Dataset tab

The Dataset tab reports, for both train and validation splits:

- image files found on disk;
- image entries in COCO `annotations.json`;
- annotation count;
- category count;
- annotation count by class.

It also warns about missing split folders, missing/invalid annotation files, image-count mismatches, labels with no annotations, and class imbalance above 10:1.

These are structural checks. They do not replace visual review of annotation quality.

## Live training tab

EPD-4 runs the existing maskrcnn-benchmark command through a streamed docker process so the GUI can observe output without waiting for the process to finish.

When the trainer emits the values, Training Studio shows:

- current iteration / configured maximum iteration;
- training loss;
- learning rate;
- ETA;
- validation AP;
- recent raw trainer output.

A missing metric is shown as unavailable rather than estimated.

### Understanding the guidance

Training loss going down is useful, but it does not prove that the model generalizes. The legacy trainer does not consistently emit validation loss, so Training Studio does not claim that falling training loss means there is no overfitting.

Prefer validation AP, inspection of validation images and real deployment tests when selecting a checkpoint.

## Checkpoints and recovery

The Checkpoints & Export tab lists `.pth` files from:

- the current `weights/custom/` directory;
- archived `weights/archived-on-*` runs.

`last_checkpoint` is used to identify the current latest checkpoint when the trainer created that marker.

### Resume selected

Select a checkpoint and choose **Resume selected**. The next Train action preserves the active `weights/custom/` directory and loads the selected checkpoint through the maskrcnn-benchmark `MODEL.WEIGHT` override.

### Resume latest

Choose **Resume latest** to use the current `last_checkpoint` marker where available, otherwise the newest current checkpoint.

### Fresh run

Choose **Fresh run** to clear the resume request. The next Train action restores the previous EPD behaviour: the current `weights/custom/` directory is archived before a new run begins.

## Selecting the model to export

Resume selection and export selection are deliberately separate.

Select a checkpoint and choose **Use selected for export** when that is the checkpoint you want converted to ONNX. This avoids assuming that the final checkpoint is automatically the best one.

Choose **Export selected now** to run the existing exporter without starting another training run.

After export, EPD-4 places the generated ONNX file in `data/model/` and asks the EPD-3 Smart Model Manager inspector to validate the model/label/mode compatibility.

## Stop training

**Stop training** terminates the active host-side docker exec process and sends an interrupt to `tools/train_net.py` in the dedicated trainer container.

Treat interruption as a recovery action. Refresh checkpoints afterward and confirm that the checkpoint you plan to resume from was completely written.

## Requirements and limitations

- The existing P2/P3 training backend still requires the NVIDIA/CUDA trainer environment used by EPD.
- Docker setup, training dependencies and exporter dependencies remain owned by the existing Trainer classes.
- Validation AP is displayed only if the underlying trainer/evaluator emits it.
- EPD-4 does not fabricate validation loss or choose a statistically "best" checkpoint without validation evidence.
- Resume relies on the checkpoint being readable by the existing maskrcnn-benchmark checkpointer.
- The Training Studio changes training observability only; it does not change perception inference algorithms, ROS message schemas, Workcell Studio ownership or robot-motion safety.
