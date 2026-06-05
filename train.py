"""Train CfC-100 on the preprocessed CHB-MIT segments.

Run:  python train.py
      (background) python train.py > logs/training.log 2>&1 &
Monitor:  tail -f logs/training.log
"""

import json

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import config
from model import CfCSeizurePredictor


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

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE,
        sampler=_weighted_sampler(labels_np[train_idx]), num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    model = CfCSeizurePredictor()

    checkpoint = ModelCheckpoint(
        dirpath=str(config.MODEL_DIR),
        filename="cfc100-{epoch:02d}-{val_auc:.3f}",
        monitor="val_auc", mode="max", save_top_k=1, save_last=True,
    )
    early_stop = EarlyStopping(monitor="val_auc", mode="max", patience=20)

    trainer = Trainer(
        max_epochs=config.MAX_EPOCHS,
        callbacks=[checkpoint, early_stop],
        accelerator="auto", devices=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0,                 # guards against CfC NaN divergence
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        default_root_dir=str(config.LOG_DIR),
    )

    trainer.fit(model, train_loader, val_loader)
    print("Training complete. Best model:", checkpoint.best_model_path)


if __name__ == "__main__":
    main()
