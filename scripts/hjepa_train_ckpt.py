"""Issue-123 Tier T launcher: the official ``main_hjepa.py`` with ViT activation checkpointing and periodic GC.

Usage (hjepa env, cwd = <H-JEPA>/h_jepa): python <repo>/scripts/hjepa_train_ckpt.py <main_hjepa.py hydra args>

The paper trains at batch 128 on one H200 (141 GB). On one RTX 5090 (32 GB) the level-1 ViT activations of
a 128-clip batch (up to 12 encoded frames per clip at 3 levels) do not fit. This launcher only

1. turns on the Hugging Face ViT's non-reentrant activation checkpointing on every ViT the official
   ``create_hf_vit`` builds. Checkpointing recomputes the forward pass in the backward pass; the ViT has no
   dropout (hidden/attention dropout 0.0), so gradients are those of the uncheckpointed model and the batch
   size, SIGReg sample and BatchNorm statistics stay those of the paper;
2. runs ``gc.collect()`` every 10 training batches (a Lightning callback appended to the official trainer's
   callbacks). A reference cycle per step keeps that step's autograd graph alive until a generation-2
   collection, which becomes rare as long-lived objects accumulate; measured on fourroom_hjepa_l3 seed 43, the
   allocated memory grew 1.8 GB per step from step ~555 to a CUDA OOM at step 578 without it, and stayed flat
   (peak <= 8.8 GiB) over 1300 steps with it. Collection frees memory only; no number changes;
3. stops a run on SIGTERM (GPU jobs only run 18:00-07:00 Melbourne; the night scheduler sends SIGTERM at 06:50,
   SIGKILL at 07:00) and exits with code 75 ("paused"; relaunched the next night). Lightning only flags the
   signal; at the end of the current train batch, after the official callbacks, the official
   ``ResumeCheckpoint`` writes the full resume state with its own ``_save`` (same file, format and atomic
   replace as its periodic saves). The relaunch resumes through the official ``ResumableManager`` as after
   any preemption: global step, optimizer and LR schedule continue exactly and the interrupted epoch runs
   only its remaining batches, but the dataloader is not fast-forwarded: the official loader re-seeds its
   generator at launch, so the remaining batches start a new shuffle sequence (they replay the beginning of
   the run's first-epoch order) and the sample order differs from an uninterrupted run.
   A stop at an epoch's last batch skips that epoch's validation and epoch-end hooks on resume. A SIGTERM
   during validation or the epoch-end hooks makes Lightning stop by itself (``SIGTERMException``, a bare
   ``SystemExit`` that would exit 0) at the last periodic or epoch-end resume file; that also exits 75. A
   SIGTERM during the train-end planning eval does not stop it (SIGKILL at 07:00 does; its finished per-task
   chunks stay on disk). A relaunch from a resume file whose training is finished (stopped in the final
   epoch-end hooks, or killed in the train-end work) makes Lightning skip the fit loop and with it the
   official ``on_train_end`` hooks (final object dump, planning eval); the launcher then runs that skipped
   train-end call itself (``fit_loop.on_run_end``: callbacks, module, strategy, in Lightning's order) while
   the model is still on the training device, so the planning eval resumes from its chunks and runs once.

then runs ``main_hjepa.py`` unchanged.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
import runpy
import sys

H_JEPA = Path(os.environ.get("HJEPA_REF", "/mnt/array/sukaih/hjepa-ref/H-JEPA")) / "h_jepa"
GC_EVERY = 10
PAUSED = 75  # exit code of a run stopped by SIGTERM after saving its resume checkpoint


def main():
    sys.path.insert(0, str(H_JEPA))
    os.chdir(H_JEPA)
    import lightning.pytorch as pl
    from lightning.pytorch.utilities.exceptions import SIGTERMException
    import models.encoders.build_encoder as build_encoder
    import models.encoders.hf_vit as hf_vit
    from utils import ResumeCheckpoint

    official = hf_vit.create_hf_vit

    def create_checkpointed_vit(*args, **kwargs):
        model = official(*args, **kwargs)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return model

    hf_vit.create_hf_vit = create_checkpointed_vit
    build_encoder.create_hf_vit = create_checkpointed_vit

    class PeriodicGC(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if batch_idx % GC_EVERY == 0:
                gc.collect()

    class StopOnSigterm(pl.Callback):
        trained = False  # whether this launch ran the fit loop (Lightning skips it when training is finished)

        def on_train_start(self, trainer, pl_module):
            self.trained = True

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.received_sigterm:
                (resume,) = [c for c in trainer.callbacks if isinstance(c, ResumeCheckpoint)]
                resume._save(trainer)
                raise SystemExit(PAUSED)

    official_trainer = pl.Trainer.__init__

    def trainer_with_gc(self, *args, **kwargs):
        stop = StopOnSigterm()
        kwargs["callbacks"] = [*(kwargs.get("callbacks") or []), PeriodicGC(), stop]
        official_trainer(self, *args, **kwargs)
        official_fit_loop_run = self.fit_loop.run

        def fit_loop_run():
            official_fit_loop_run()
            if not stop.trained and self.current_epoch >= self.max_epochs:
                self.fit_loop.on_run_end()  # the train-end hooks Lightning skips for a finished resume file

        self.fit_loop.run = fit_loop_run

    pl.Trainer.__init__ = trainer_with_gc
    sys.argv = [str(H_JEPA / "main_hjepa.py"), *sys.argv[1:]]
    try:
        runpy.run_path(str(H_JEPA / "main_hjepa.py"), run_name="__main__")
    except SIGTERMException:
        raise SystemExit(PAUSED) from None


if __name__ == "__main__":
    main()
