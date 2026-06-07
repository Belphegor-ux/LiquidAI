import argparse
import json
import sys

import numpy as np
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

import config
from evaluate import (
    _fa_per_hour,
    _per_seizure_sensitivity,
    _subset_tensor,
    _tune_threshold,
    find_checkpoint,
    predict_in_batches,
)
from model import CfCSeizurePredictor


def main():
    parser = argparse.ArgumentParser(description="Patient-specific fine-tuning")
    parser.add_argument("patient_id", type=str, help="Patient ID (e.g., chb01)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning")
    args = parser.parse_args()

    patient_id = args.patient_id

    # 1. Load data
    segments_mm = np.load(config.OUTPUT_DIR / "segments.npy", mmap_mode="r")
    labels = np.load(config.OUTPUT_DIR / "labels.npy")

    with open(config.OUTPUT_DIR / "patient_ids.json") as f:
        patient_ids = np.array(json.load(f), dtype=object)
    with open(config.OUTPUT_DIR / "seizure_groups.json") as f:
        groups = np.array(json.load(f), dtype=object)

    # 2. Filter for patient
    patient_mask = patient_ids == patient_id
    if not np.any(patient_mask):
        print(f"Error: Patient {patient_id} not found.")
        sys.exit(1)

    patient_idx = np.where(patient_mask)[0]
    
    # Chronological split: first 20% for train, 80% for eval
    split_point = int(len(patient_idx) * 0.2)
    if split_point == 0:
        print("Not enough data for 20% split.")
        sys.exit(1)

    train_idx = patient_idx[:split_point]
    test_idx = patient_idx[split_point:]

    print(f"Patient {patient_id}: {len(patient_idx)} total epochs")
    print(f"Fine-tuning on first {len(train_idx)} epochs (20%)")
    print(f"Evaluating on remaining {len(test_idx)} epochs (80%)")

    # Load segments into memory
    train_segments = _subset_tensor(segments_mm, train_idx)
    train_labels = torch.from_numpy(labels[train_idx])
    train_ds = TensorDataset(train_segments, train_labels)

    # Use shuffle=False if you want to preserve chronological order in training, 
    # but shuffle=True is usually better for SGD stability.
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    # 3. Load Model
    ckpt = find_checkpoint()
    print(f"Loading zero-shot checkpoint: {ckpt}")
    
    model = CfCSeizurePredictor.load_from_checkpoint(ckpt)
    
    # Adjust learning rate for fine-tuning
    model.hparams.learning_rate = args.lr

    # 4. Fine-tune
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=5,
        enable_checkpointing=False,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
    )
    
    print("\nStarting fine-tuning...")
    trainer.fit(model, train_loader)
    
    save_path = config.MODEL_DIR / f"{patient_id}_finetuned.ckpt"
    trainer.save_checkpoint(save_path)
    print(f"Fine-tuning complete. Saved personalized model to {save_path}\n")

    # 5. Evaluate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    test_labels = labels[test_idx]
    test_segments = _subset_tensor(segments_mm, test_idx)
    test_preds = predict_in_batches(model, test_segments, device)
    
    test_groups = groups[test_idx]
    
    # Tune threshold on the 20% train data
    train_preds = predict_in_batches(model, train_segments, device)
    thr = _tune_threshold(train_labels.numpy(), train_preds, config.EVAL_TARGET_FA_PER_H)
    print(f"Tuned threshold on 20% data: {thr:.3f}")

    both_classes = len(np.unique(test_labels)) > 1
    auroc = float(roc_auc_score(test_labels, test_preds)) if both_classes else float("nan")
    auprc = float(average_precision_score(test_labels, test_preds)) if both_classes else float("nan")
    
    cm = confusion_matrix(test_labels, (test_preds >= thr).astype(int), labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    win_sens = tp / (tp + fn) if (tp + fn) else 0.0
    win_spec = tn / (tn + fp) if (tn + fp) else 0.0
    fa_h = _fa_per_hour(test_labels, test_preds, thr)

    seiz_sens, hit, total = _per_seizure_sensitivity(test_labels, test_preds, test_groups, thr)

    print(f"\n== Evaluation on 80% Held-out Data for Patient {patient_id} ==")
    print(f"Per-seizure sensitivity: {seiz_sens:.3f}  ({hit}/{total} seizures)")
    print(f"False alarms / hour:     {fa_h:.3f}  (target < {config.EVAL_TARGET_FA_PER_H})")
    print(f"AUROC: {auroc:.3f} | AUPRC: {auprc:.3f}")
    print(f"Per-window sensitivity: {win_sens:.3f} | specificity: {win_spec:.3f}")
    print(f"Confusion [[tn fp][fn tp]]:\n{cm}")


if __name__ == "__main__":
    main()
