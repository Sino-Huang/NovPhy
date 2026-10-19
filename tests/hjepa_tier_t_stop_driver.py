"""Toy stand-in for the official ``main_hjepa.py`` (tests/test_hjepa_tier_t_stop.py; hjepa env, CPU).

The test links it as ``<HJEPA_REF>/h_jepa/main_hjepa.py`` next to the official ``utils.py`` and ``models/`` and
starts it through ``scripts/hjepa_train_ckpt.py``, so the launcher's Trainer patch, callbacks and exit handling
are the real ones. Like ``main_hjepa.py`` it builds the trainer with the official ``utils.ResumeCheckpoint`` first,
a seeded shuffled DataLoader, a step-wise LR schedule and one validation pass per epoch, and resumes from
``<run_dir>/lightning_resume/last.ckpt`` when it exists. Every train/val batch and hook is logged to
``<run_dir>/seen.jsonl``.

Usage: main_hjepa.py RUN_DIR [--sigterm train:K|val:EPOCH|epoch_end:EPOCH|train_end] [--sigkill-train-end]
                     [--every-n N] [--epochs E]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import ResumeCheckpoint

SAMPLES, BATCH = 24, 2  # 12 batches per epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--sigterm", default="")
    parser.add_argument("--sigkill-train-end", action="store_true")
    parser.add_argument("--every-n", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    where, _, at = args.sigterm.partition(":")
    log_path = args.run_dir / "seen.jsonl"
    resume_file = args.run_dir / "lightning_resume" / "last.ckpt"

    def note(**entry):
        with log_path.open("a") as handle:
            handle.write(json.dumps({"pid": os.getpid(), **entry}) + "\n")

    def sigterm_self():
        os.kill(os.getpid(), signal.SIGTERM)

    class Toy(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)

        def training_step(self, batch, batch_idx):
            (x,) = batch
            step = self.trainer.global_step
            note(kind="train", epoch=self.current_epoch, batch_idx=batch_idx, global_step=step,
                 lr=self.trainer.optimizers[0].param_groups[0]["lr"], idx=x[:, 0].long().tolist())
            if where == "train" and step == int(at):
                sigterm_self()
            return self.layer(x).pow(2).mean()

        def validation_step(self, batch, batch_idx):
            if self.trainer.sanity_checking:
                return
            note(kind="val", epoch=self.current_epoch, batch_idx=batch_idx, global_step=self.trainer.global_step)
            if where == "val" and batch_idx == 0 and self.current_epoch == int(at):
                sigterm_self()

        def on_train_epoch_end(self):
            note(kind="epoch_end", epoch=self.current_epoch, global_step=self.trainer.global_step)
            if where == "epoch_end" and self.current_epoch == int(at):
                sigterm_self()

        def on_train_end(self):
            note(kind="train_end", epoch=self.current_epoch, global_step=self.trainer.global_step,
                 device=str(self.device))
            if where == "train_end":
                sigterm_self()
                note(kind="train_end_done", global_step=self.trainer.global_step)

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=1.0)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1 + step))
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    class TrainEndWork(pl.Callback):  # stands in for the official planning eval run at train end
        def on_train_end(self, trainer, pl_module):
            note(kind="train_end_work", epoch=trainer.current_epoch, global_step=trainer.global_step)
            if args.sigkill_train_end:
                os.kill(os.getpid(), signal.SIGKILL)

    data = TensorDataset(torch.arange(SAMPLES, dtype=torch.float32)[:, None])
    train = DataLoader(data, batch_size=BATCH, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(42))
    val = DataLoader(data, batch_size=8)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=args.epochs, num_sanity_val_steps=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=[ResumeCheckpoint(resume_file, args.every_n), TrainEndWork()],
    )
    resuming = resume_file.is_file()
    note(kind="start", resume=resuming)
    trainer.fit(Toy(), train, val, ckpt_path=str(resume_file) if resuming else None)
    note(kind="end", global_step=trainer.global_step, epoch=trainer.current_epoch)


if __name__ == "__main__":
    main()
