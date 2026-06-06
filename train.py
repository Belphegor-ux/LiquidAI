"""Train CfC-100 on the preprocessed CHB-MIT segments.

Run:  python train.py
      (background) python train.py > logs/training.log 2>&1 &
Monitor:  tail -f logs/training.log
"""

import json
import os
import sys

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import config
from model import CfCSeizurePredictor

# Windows consoles may use a non-UTF-8 code page (e.g. cp932); force UTF-8 so the
# Lightning Rich progress bar's Unicode glyphs don't crash the run with
# UnicodeEncodeError. errors="replace" guards any remaining odd characters.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Use Tensor-Core TF32 matmuls on the RTX 3050 -- a free speedup for the dense
# layers; the CfC recurrence is the main cost but every bit helps.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


def _resume_ckpt():
    """Latest 'last' checkpoint to resume from, or None.

    Opt in with `python train.py --resume` or NEUROFLOW_RESUME=1; restores epoch,
    optimizer state and callbacks so a paused run continues instead of restarting.
    """
    if "--resume" not in sys.argv and os.environ.get("NEUROFLOW_RESUME") != "1":
        return None
    candidates = sorted(config.MODEL_DIR.glob("last*.ckpt"), key=lambda p: p.stat().st_mtime)
    return str(candidates[-1]) if candidates else None


def _weighted_sampler(labels):
    """Balance the heavy interictal/preictal imbalance during training."""
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_weights = 1.0 / np.clip(class_counts, 1.0, None)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def main():
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    segments = torch.from_numpy(np.load(config.OUTPUT_DIR / "segments.npy")).float()
    labels_np = np.load(config.OUTPUT_DIR / "labels.npy").astype(np.int64)
    labels = torch.from_numpy(labels_np)

    with open(config.OUTPUT_DIR / "split.json") as f:
        split = json.load(f)

    train_idx, val_idx = split["train_idx"], split["val_idx"]
    train_ds = TensorDataset(segments[train_idx], labels[train_idx])
    val_ds = TensorDataset(segments[val_idx], labels[val_idx])

    if config.USE_WEIGHTED_SAMPLER:
        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            sampler=_weighted_sampler(labels_np[train_idx]),
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
        )
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    model = CfCSeizurePredictor()

    # Monitor val_loss (focal): defined on every batch and stable, unlike val_auc
    # which is rank-noisy on small val sets.
    checkpoint = ModelCheckpoint(
        dirpath=str(config.MODEL_DIR),
        filename="cfc100-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [checkpoint]
    if config.EARLY_STOP_PATIENCE > 0:
        callbacks.append(
            EarlyStopping(monitor="val_loss", mode="min", patience=config.EARLY_STOP_PATIENCE)
        )
    else:
        print("Early stopping disabled; training full", config.MAX_EPOCHS, "epochs.")

    trainer = Trainer(
        max_epochs=config.MAX_EPOCHS,
        callbacks=callbacks,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # guards against CfC NaN divergence
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        default_root_dir=str(config.LOG_DIR),
    )

    ckpt_path = _resume_ckpt()
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    print("Training complete. Best model:", checkpoint.best_model_path)


if __name__ == "__main__":
    main()
