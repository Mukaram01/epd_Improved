"""EPD-4 productization entry point."""

from pathlib import Path
import shutil
import subprocess

from types import MethodType

from windows.training_studio import (
    TrainingStudioController,
    _copy_with_privilege_fallback,
    apply_training_studio,
)


def _safe_stage_dataset(self):
    source = Path(self.train._path_to_dataset).expanduser().resolve()
    staging = (self.gui_dir / "trainer/training_files/custom_dataset").resolve()
    if source == staging:
        return staging
    if staging.exists():
        try:
            shutil.rmtree(staging)
        except OSError:
            subprocess.run(["sudo", "rm", "-rf", str(staging)], check=True)
    shutil.copytree(source, staging)
    return staging


def _safe_start_training(self):
    """Run training in the worker thread without touching Qt widgets directly."""
    trainer = self._new_trainer()
    if not getattr(trainer, "isGPUAvailableFlag", False):
        raise RuntimeError(
            "Training requires the GPU/CUDA trainer environment; "
            "nvidia-smi or nvcc failed."
        )

    self.history = []
    self.current_iteration = 0
    self.current_loss = None
    self.current_lr = None
    self.current_eta = None
    self.current_ap = None
    self._last_onnx_mtime = self._newest_onnx_mtime()
    self.events.put(("line", "--- New Training Studio run ---"))

    trainer.copyTrainingFiles = MethodType(
        lambda instance: self._copy_training_files(instance),
        trainer,
    )
    trainer.runTraining = MethodType(
        lambda instance: self._run_training_process(instance),
        trainer,
    )
    trainer.runExporter = MethodType(
        lambda instance: self._run_exporter_process(instance),
        trainer,
    )

    self.events.put(("phase", "TRAINING"))
    trainer.train(False)

    selected = self.export_checkpoint
    if selected:
        _copy_with_privilege_fallback(selected, self.gui_dir / "trained.pth")
    trainer.export(False)
    self._validate_latest_onnx()
    self.resume_checkpoint = None
    self.events.put(("phase", "COMPLETE"))


def apply_epd4_productization(main_window):
    """Install the EPD-4 Training Studio on the current launcher."""
    TrainingStudioController._stage_dataset = _safe_stage_dataset
    TrainingStudioController._start_training = _safe_start_training
    return apply_training_studio(main_window)
